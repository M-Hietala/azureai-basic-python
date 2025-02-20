'''
Created on Feb 13, 2025

@author: nirovins
'''
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch
from azure.identity.aio import DefaultAzureCredential

from routes import ChatRequest, Message
from rag_helper import RAGHelper
from azure.ai.projects.aio import AIProjectClient
from azure.core.exceptions import ResourceNotFoundError


class MockAsyncIterator:
    
    def __init__(self, list_data):
        assert list_data and isinstance(list_data, list)
        self._data = list_data

    async def __aiter__(self):
        for dt in self._data:
            yield dt


class TestRAGHelper(unittest.IsolatedAsyncioTestCase):
    """Tests for the RAG helper."""

    INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
    EMBEDDINGS_FILE = os.path.join(INPUT_DIR, 'embeddings.csv')
    
    @classmethod
    def setUpClass(cls)->None:
        super(TestRAGHelper, cls).setUpClass()

    def setUp(self)->None:
        self.search_endpoint = "https://nirovins-westus-search.search.windows.net"
        self.index_name = "test_index"
        unittest.TestCase.setUp(self)

    async def test_index_exist_mock(self):
        """Test index exists check."""
        mock_ix_client = AsyncMock()
        mock_aenter = AsyncMock()
        with patch('rag_helper.SearchIndexClient', return_value=mock_ix_client):
            mock_ix_client.__aenter__.return_value = mock_aenter
            exists = await RAGHelper.index_exists(self.search_endpoint, AsyncMock(), self.index_name)
            self.assertTrue(exists)
            mock_aenter.get_index.side_effect = ResourceNotFoundError("Mock")
            exists = await RAGHelper.index_exists(self.search_endpoint, AsyncMock(), self.index_name)
            self.assertFalse(exists)
        

    async def test_get_or_create_mock(self):
        """Test index_name creation."""
        mock_ix_client = AsyncMock()
        mock_aenter = AsyncMock()
        with patch('rag_helper.SearchIndexClient', return_value=mock_ix_client):
            mock_ix_client.__aenter__.return_value = mock_aenter
            mock_aenter.get_index.side_effect = ResourceNotFoundError("Mock")
            await RAGHelper.get_or_create_search_index(
                endpoint=self.search_endpoint,
                credential=AsyncMock(),
                index_name=self.index_name,
                dimensions=100)
            mock_aenter.create_index.assert_called_once()
            mock_aenter.create_index.reset_mock()
            mock_aenter.get_index.assert_called_once()
            mock_aenter.get_index.reset_mock()
            mock_aenter.get_index.side_effect = Mock()
            await RAGHelper.get_or_create_search_index(
                endpoint=self.search_endpoint,
                credential=AsyncMock(),
                index_name=self.index_name,
                dimensions=100500)
            mock_aenter.get_index.assert_called_once()
            mock_aenter.create_index.assert_not_called()

    async def test_no_index_exception(self):
        """Test that attempt to search without index causes the exception."""
        rag = self._get_mock_rag(AsyncMock())
        with self.assertRaisesRegex(ValueError, "Unable to perform the operation as the index is absent.+"):
            await rag.delete_index()
        with self.assertRaisesRegex(ValueError, "Unable to perform the operation as the index is absent.+"):
            await rag.upload_documents("test.csv")
        with self.assertRaisesRegex(ValueError, "Unable to perform the operation as the index is absent.+"):
            await rag.search(ChatRequest(messages=[Message(content='test')]))
        with self.assertRaisesRegex(ValueError, "Unable to perform the operation as the index is absent.+"):
            await rag.is_index_empty()

    async def test_create_delete_mock(self):
        """Test that if index is deleteed the appropriate error is raised."""
        mock_ix_client = AsyncMock()
        mock_aenter = AsyncMock()
        with patch('rag_helper.SearchIndexClient', return_value=mock_ix_client):
            mock_ix_client.__aenter__.return_value = mock_aenter
            rag = self._get_mock_rag(AsyncMock())
            await rag.create_index_maybe()
            mock_aenter.get_index.assert_called_once()
            mock_aenter.get_index.reset_mock()
            await rag.create_index_maybe()
            mock_aenter.get_index.assert_not_called()
            await rag.delete_index()
            mock_aenter.delete_index.assert_called_once()
            with self.assertRaisesRegex(ValueError, "Unable to perform the operation as the index is absent.+"):
                await rag.delete_index()

    async def test_life_cycle_mock(self):
        """Test create, upload, search and delete"""
        mock_ix_client = AsyncMock()
        mock_aenter = AsyncMock()
        mock_serch_client = AsyncMock()
        mock_search_aenter = AsyncMock()
        mock_search_aenter.search.return_value = MockAsyncIterator([
            {'token': 'a'},
            {'token': 'b'}
        ])
        mock_embedding = AsyncMock()
        mock_embedding.embed.return_value = {
            'data': [{'embedding': 42.}]
        }
        with patch('rag_helper.SearchIndexClient', return_value=mock_ix_client):
            with patch('rag_helper.SearchClient', return_value=mock_serch_client):
                mock_ix_client.__aenter__.return_value = mock_aenter
                mock_serch_client.__aenter__.return_value = mock_search_aenter
                rag = self._get_mock_rag(mock_embedding)
                await rag.create_index_maybe()
                
                # Upload documents.
                await rag.upload_documents(TestRAGHelper.EMBEDDINGS_FILE)
                mock_search_aenter.upload_documents.assert_called_once()

                message = ChatRequest(messages=[Message(content='test')])
                search_result = await rag.search(message)
                mock_embedding.embed.assert_called_once()
                mock_search_aenter.search.assert_called_once()
                self.assertEqual(search_result, "a\n------\nb")

    async def test_is_empty_mock(self):
        """Test how we check if the index is empty."""
        mock_ix_client = AsyncMock()
        mock_aenter = AsyncMock()
        mock_serch_client = AsyncMock()
        mock_search_aenter = AsyncMock()
        
        with patch('rag_helper.SearchIndexClient', return_value=mock_ix_client):
            with patch('rag_helper.SearchClient', return_value=mock_serch_client):
                mock_ix_client.__aenter__.return_value = mock_aenter
                mock_serch_client.__aenter__.return_value = mock_search_aenter
                mock_search_aenter.get_document_count.return_value = 42
                rag = self._get_mock_rag(AsyncMock())
                await rag.create_index_maybe()
                is_empty = await rag.is_index_empty()
                self.assertFalse(is_empty)
                mock_search_aenter.get_document_count.return_value = 0
                is_empty = await rag.is_index_empty()
                self.assertTrue(is_empty)

    async def test_exception_no_dinmensions(self):
        """Test the exception shown if no dimensions were provided."""
        rag = RAGHelper(
            endpoint=self.search_endpoint,
            credential=AsyncMock(),
            index_name=self.index_name,
            dimensions=None,
            model="mock_embedding_model",
            embeddings_client=AsyncMock()
        )
        with self.assertRaisesRegex(ValueError, "No embedding dimensions were provided.+"):
            await rag.create_index_maybe(dimensions_override=None)

    async def test_exception_different_dinmensions(self):
        """Test the exception shown if dimensions and dinensions_override are different."""
        rag = RAGHelper(
            endpoint=self.search_endpoint,
            credential=AsyncMock(),
            index_name=self.index_name,
            dimensions=41,
            model="mock_embedding_model",
            embeddings_client=AsyncMock()
        )
        with self.assertRaisesRegex(ValueError, "dimensions_override is different from dimensions provided to constructor."):
            await rag.create_index_maybe(dimensions_override=42)

    @unittest.skip("Only for live tests.")
    async def test_e2e(self):
        """Run search end to end."""
        async with DefaultAzureCredential() as creds:
            async with AIProjectClient.from_connection_string(
                credential=creds,
                conn_str=os.environ["AZURE_AIPROJECT_CONNECTION_STRING"],
            ) as project: 
                async with (await project.inference.get_embeddings_client()) as embed:
                    rag = RAGHelper(
                        endpoint=self.search_endpoint,
                        credential=creds,
                        index_name=self.index_name,
                        dimensions=100,
                        model="text-embedding-3-small",
                        embeddings_client=embed,
                    )
                    await rag.create_index_maybe()
                    await rag.upload_documents(TestRAGHelper.EMBEDDINGS_FILE)
                    
                    result = await rag.search(
                        ChatRequest(
                            messages=[
                                Message(content="What is the temperature rating of the cozynights sleeping bag?")
                            ]
                        )
                    )
                    await rag.delete_index()
                    self.assertTrue(bool(result))
                

    @unittest.skip("Only for live tests.")
    async def test_build_embeddings_file(self):
        """Use this test to build the new embeddings file in the data directory."""
        
        async with DefaultAzureCredential() as creds:
            async with AIProjectClient.from_connection_string(
                credential=creds,
                conn_str=os.environ["AZURE_AIPROJECT_CONNECTION_STRING"],
            ) as project:
                async with (await project.inference.get_embeddings_client()) as embed:
                    rag = RAGHelper(
                        endpoint=self.search_endpoint,
                        credential=creds,
                        index_name=self.index_name,
                        dimensions=100,
                        model="text-embedding-3-small",
                        embeddings_client=embed,
                    )
                    await rag.build_embeddings_file(
                        input_directory=TestRAGHelper.INPUT_DIR,
                        output_file=TestRAGHelper.EMBEDDINGS_FILE)
        

    @unittest.skip("Only for live tests.")
    async def test_get_or_create(self):
        """Test index_name creation."""
        async with DefaultAzureCredential() as cred:
            #/with patch('RAGHelper._get_search_index_client') as mock_ix_client:
            await RAGHelper.get_or_create_search_index(
                endpoint=self.search_endpoint,
                credential=cred,
                index_name=self.index_name,
                dimensions=100)
            await RAGHelper.get_or_create_search_index(
                endpoint=self.search_endpoint,
                credential=cred,
                index_name=self.index_name,
                dimensions=100500)

    def _get_mock_rag(self, embedding_client):
        """Return the mock RAG """
        return RAGHelper(
            endpoint=self.search_endpoint,
            credential=AsyncMock(),
            index_name=self.index_name,
            dimensions=100,
            model="mock_embedding_model",
            embeddings_client=embedding_client
        )

    # async def asyncTearDown(self)->None:
    #     async with DefaultAzureCredential() as cred:
    #         async with SearchIndexClient(endpoint=self.search_endpoint, credential=cred) as ix_client:
    #             try:
    #                 await ix_client.delete_index(self.index_name)
    #             except ResourceNotFoundError:
    #                 pass
    #     unittest.TestCase.tearDown(self)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()