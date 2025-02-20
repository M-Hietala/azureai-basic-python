from typing import Optional

import glob
import csv
import json

from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery 
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,  
    SimpleField,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration)
from azure.ai.inference.aio import EmbeddingsClient
from azure.core.exceptions import ResourceNotFoundError
from .routes import ChatRequest



class RAGHelper:
    """
    The singletone class for searching of context for user queries.

    :param endpoint: The search endpoint to be used.
    :param credential: The credential to be used for the search.
    :param index_name: The name of an index to get or to create.
    :param dimensions: The number of dimensions in the embedding. Set this parameter only if
                       embedding model accepts dimensions parameter.
    :param model: The embedding model to be used,
                  must be the same as one use to build the file with embeddings.
    :param embeddings_client: The embedding client.
    """
    
    def __init__(
            self,
            endpoint: str,
            credential: AsyncTokenCredential,
            index_name: str,
            dimensions: Optional[int],
            model: str,
            embeddings_client: EmbeddingsClient,
        ) -> None:
        """Constructor."""
        self._dimensions = dimensions
        self._index_name = index_name
        self._embeddings_client = embeddings_client
        self._endpoint = endpoint
        self._credential = credential
        self._index = None
        self._model = model

    async def search(self, message: ChatRequest) -> str:
        """
        Search the message in the vector store.

        :param message: The customer question.
        :return: The context for the question.
        """
        self._raise_if_no_index()
        embedded_question = (await self._embeddings_client.embed(
            input=message.messages[-1].content,
            dimensions=self._dimensions,
            model=self._model
        ))['data'][0]['embedding']
        vector_query = VectorizedQuery(vector=embedded_question, k_nearest_neighbors=5, fields="embedding")
        async with SearchClient(endpoint=self._endpoint, index_name=self._index.name, credential=self._credential) as search_client:
            response = await search_client.search(
                vector_queries=[vector_query],
                select=['token'],
            )
            results = [result['token'] async for result in response]
        return "\n------\n".join(results)
    
    async def upload_documents(self, embeddings_file: str) -> None:
        """
        Upload the embeggings file to index search.

        :param embeddings_file: The embeddings file to upload.
        """
        self._raise_if_no_index()
        documents = []
        index = 0
        with open(embeddings_file, newline='') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                documents.append(
                    {
                        'embedId': str(index),
                        'token': row['token'],
                        'embedding': json.loads(row['embedding'])
                    }
                )
                index += 1
        async with SearchClient(endpoint=self._endpoint, index_name=self._index.name, credential=self._credential) as search_client:
            await search_client.upload_documents(documents)

    async def is_index_empty(self) -> bool:
        """
        Return True if the index is empty.

        :return: True f index is empty.
        """
        if self._index is None:
            raise ValueError(
                "Unable to perform the operation as the index is absent. "
                "To create index please call create_index")
        async with SearchClient(endpoint=self._endpoint, index_name=self._index.name, credential=self._credential) as search_client:
            document_count = await search_client.get_document_count()
        return document_count == 0

    def _raise_if_no_index(self) -> None:
        """
        Raise the exception if the index was not created.

        :raises: ValueError
        """
        if self._index is None:
            raise ValueError(
                "Unable to perform the operation as the index is absent. "
                "To create index please call create_index")

    async def delete_index(self):
        """Delete the index from vector store."""
        self._raise_if_no_index()
        async with SearchIndexClient(endpoint=self._endpoint, credential=self._credential) as ix_client:
            await ix_client.delete_index(self._index.name)
        self._index = None

    async def create_index_maybe(self, dimensions_override: Optional[int] = None) -> None:
        """
        Create the index if it does not exist.

        :param dimensions_override: The number of dimensions in the embedding. This parameter is
               needed if the embedding parameter cannot be set for the given model. It can be
               figured out by loading the embeddings file, generated by build_embeddings_file,
               loading the contents of the first row and 'embedding' column as a JSON and calculating
               the length of the list obtained.
               Also please see the embedding model documentation
               https://platform.openai.com/docs/models#embeddings
        """
        if dimensions_override is None:
            if self._dimensions is None:
                raise ValueError(
                    "No embedding dimensions were provided in neither dimensions in the constructor nor in dimensions_override"
                    "Dimensions are needed to build the search index, please provide the dimensions_override.")
            dimensions_override = self._dimensions
        if self._dimensions is not None and dimensions_override != self._dimensions:
            raise ValueError("dimensions_override is different from dimensions provided to constructor.")
        
        if self._index is None:
            self._index = await RAGHelper.get_or_create_search_index(
                self._endpoint,
                self._credential,
                self._index_name,
                dimensions_override)

    @staticmethod
    async def index_exists(
        endpoint: str,
        credential: AsyncTokenCredential,
        index_name: str) -> bool:
        """
        Check if index exists.

        :param endpoint: The search end point to be used.
        :param credential: The credential to be used for the search.
        :param index_name: The name of an index to get or to create.
        :return: True if index already exists.
        """
        exists = False
        async with SearchIndexClient(endpoint=endpoint, credential=credential) as ix_client:
            try:
                await ix_client.get_index(index_name)
                exists = True
            except ResourceNotFoundError:
                pass
        return exists

    @staticmethod
    async def get_or_create_search_index(
            endpoint: str,
            credential: AsyncTokenCredential,
            index_name: str,
            dimensions: int,
        ) -> SearchIndex:
        """
        Get o create the search index.

        **Note:** If the search index with index_name exists, the embeddings_file will not be uploaded.
        :param endpoint: The search end point to be used.
        :param credential: The credential to be used for the search.
        :param index_name: The name of an index to get or to create.
        :param dimensions: The number of dimensions in the embedding.
        :return: the seach index object.
        """
        #ix_client = RAGHelper._get_search_index_client(endpoint, credential)
        async with SearchIndexClient(endpoint=endpoint, credential=credential) as ix_client:
            try:
                index = await ix_client.get_index(index_name)
            except ResourceNotFoundError:
                fields = [
                    SimpleField(name="embedId", type=SearchFieldDataType.String, key=True),
                    SearchField(
                        name="embedding",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        vector_search_dimensions=dimensions,
                        searchable=True,
                        vector_search_profile_name="embedding_config"
                    ),
                    SimpleField(name="token", type=SearchFieldDataType.String, hidden=False),
                ]
                vector_search = VectorSearch(
                    profiles=[VectorSearchProfile(name="embedding_config",
                                                  algorithm_configuration_name="embed-algorithms-config")],
                    algorithms=[HnswAlgorithmConfiguration(name="embed-algorithms-config")],
                )
                search_index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
                index = await ix_client.create_index(search_index)
        return index

    async def build_embeddings_file(
            self,
            input_directory: str,
            output_file: str
            ) -> None:
        """
        In this method we do lazy loading of nltk and download the needed data set to split

        document into tokens. This operation takes time that is why we hide import nltk under this
        method. We also do not include nltk into requirements because this method is only used
        during rag generation.
        :param dimensions: The number of dimensions in the embeddings. Must be the same as
               the one used for RAGHelper creation.
        :param input_directory: The directory with the embedding files.
        :param output_file: The file csv file to store embeddings.
        :param embeddings_client: The embedding client, used to create embeddings. 
                Must be the same as the one used for RAGHelper creation.
        :param model: The embedding model to be used.
        """
        import nltk
        nltk.download('punkt')
        
        from nltk.tokenize import sent_tokenize
        # Split the data to sentence tokens.
        sentence_tokens = []
        globs = glob.glob(input_directory + '/*.md', recursive=True)
        for fle in globs:
            with open(fle) as f:
                for line in f:
                    line = line.strip()
                    sentence_tokens.extend(sent_tokenize(line))
        
        # For each token build the embedding, which will be used in the search.
        batch_size = 2000
        with open(output_file, 'w') as fp:
            writer = csv.DictWriter(fp, fieldnames=['token', 'embedding'])
            writer.writeheader()
            for i in range(0, len(sentence_tokens), batch_size):
                emedding = (await self._embeddings_client.embed(
                    input=sentence_tokens[i:i+min(batch_size, len(sentence_tokens))],
                    dimensions=self._dimensions,
                    model=self._model
                ))["data"]
                for token, float_data in zip(sentence_tokens, emedding):
                    writer.writerow({'token': token, 'embedding': json.dumps(float_data['embedding'])})
