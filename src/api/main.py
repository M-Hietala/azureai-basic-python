import contextlib
import logging
import os
import pathlib
from typing import Union

import fastapi
from azure.ai.projects.models import ConnectionType
from azure.ai.projects.aio import AIProjectClient
from azure.ai.inference.prompts import PromptTemplate
from azure.identity import AzureDeveloperCliCredential, ManagedIdentityCredential
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

from .shared import globals
from .rag_helper import RAGHelper

logger = logging.getLogger("azureaiapp")
logger.setLevel(logging.INFO)

# Configure logging to file, if log file name is provided
log_file_name = os.getenv("APP_LOG_FILE")
if log_file_name is not None:
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

enable_trace_string = os.getenv("ENABLE_AZURE_MONITOR_TRACING")
enable_trace = False
if enable_trace_string is None:
    enable_trace = False
else:
    enable_trace = str(enable_trace_string).lower() == "true"
if enable_trace:
    logger.info("Tracing is enabled.")
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
    except ModuleNotFoundError:
        logger.error("Required libraries for tracing not installed.")
        logger.error("Please make sure azure-monitor-opentelemetry is installed.")
        exit()
else:
    logger.info("Tracing is not enabled")

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    azure_credential: Union[AzureDeveloperCliCredential, ManagedIdentityCredential]
    if not os.getenv("RUNNING_IN_PRODUCTION"):
        if tenant_id := os.getenv("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            azure_credential = AzureDeveloperCliCredential(tenant_id=tenant_id)
        else:
            logger.info("Using AzureDeveloperCliCredential")
            azure_credential = AzureDeveloperCliCredential()
    else:
        # User-assigned identity was created and set in api.bicep
        user_identity_client_id = os.getenv("AZURE_CLIENT_ID")
        logger.info("Using ManagedIdentityCredential with client_id %s", user_identity_client_id)
        azure_credential = ManagedIdentityCredential(client_id=user_identity_client_id)

    project = AIProjectClient.from_connection_string(
        credential=azure_credential,
        conn_str=os.environ["AZURE_AIPROJECT_CONNECTION_STRING"],
    )

    if enable_trace:
        application_insights_connection_string = ""
        try:
            application_insights_connection_string = await project.telemetry.get_connection_string()
        except Exception as e:
            e_string = str(e)
            logger.error("Failed to get Application Insights connection string, error: %s", e_string)
        if not application_insights_connection_string:
            logger.error("Application Insights was not enabled for this project.")
            logger.error("Enable it via the 'Tracing' tab in your AI Foundry project page.")
            exit()
        else:
            configure_azure_monitor(connection_string=application_insights_connection_string)

    chat = await project.inference.get_chat_completions_client()
    prompt = PromptTemplate.from_prompty(pathlib.Path(__file__).parent.resolve() / "prompt.prompty")
    embed = await project.inference.get_embeddings_client()

    endpoint = None
    for conn in await project.connections.list(connection_type=ConnectionType.AZURE_AI_SEARCH):
        endpoint = conn.endpoint_url
        break
    
    rag = None
    if endpoint and os.getenv('AZURE_AI_SEARCH_INDEX_NAME') and os.getenv('AZURE_AI_EMBED_DEPLOYMENT_NAME'):
        rag = RAGHelper(
            endpoint = endpoint,
            credential = azure_credential,
            index_name = os.getenv('AZURE_AI_SEARCH_INDEX_NAME'),
            dimensions = 100,
            model=os.getenv('AZURE_AI_EMBED_DEPLOYMENT_NAME'),
            embeddings_client=embed
        )
        # Create index and upload the documents only if index does not exist.
        logger.info(f"Creating index {os.getenv('AZURE_AI_SEARCH_INDEX_NAME')}.")
        await rag.create_index_maybe(dimensions_override=100)
        if await rag.is_index_empty():
            logger.info(f"Uploading documents to {os.getenv('AZURE_AI_SEARCH_INDEX_NAME')}.")
            await rag.upload_documents(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(__file__))), 'data', 'embeddings.csv'))
        

    globals["project"] = project
    globals["chat"] = chat
    globals["embed"] = embed
    globals["rag"] = rag
    globals["rag_prompt"] = prompt
    globals["prompt"] = PromptTemplate.from_string('You are a helpful assistant')
    globals["chat_model"] = os.environ["AZURE_AI_CHAT_DEPLOYMENT_NAME"]
    globals["embed_model"] = os.environ["AZURE_AI_EMBED_DEPLOYMENT_NAME"]
    yield

    await project.close()
    await chat.close()


def create_app():
    if not os.getenv("RUNNING_IN_PRODUCTION"):
        logger.info("Loading .env file")
        load_dotenv(override=True)

    app = fastapi.FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="api/static"), name="static")

    from . import routes  # noqa

    app.include_router(routes.router)

    return app
