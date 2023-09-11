# author: @NavinKumarMNK

from qdrant_client import QdrantClient
from qdrant_client import grpc
import yaml
import dotenv
import os
from typing import List
from utils import generate_id

class VectorDB():
    def __init__(self):
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        self.config = config['vector-db']
        path = self.config['credentials']
        dotenv.load_dotenv(path)
        self.client = QdrantClient(
            os.getenv('URL'),
            os.getenv('API_KEY'),
        )
    
    def get_db_info(self):
        # can be used to modify custom db
        return {
            'url': os.getenv('URL'),
            'api_key': os.getenv('API_KEY'),
            'collection_name': os.getenv('COLLECTION_NAME'),
        }
    
    async def create_collections(self):
        # quadrant collection : env variable
        params = {
            'collection_name' : os.getenv('COLLECTION_NAME'),
            'vectors_config': grpc.VectorsConfig(
                params = grpc.VectorParams(
                    size=self.config['dim'],
                    distance=grpc.Distance.Cosine,
                )
            ),
            "timeout": 10
        }
        
        if self.config['quantization'] == 'int8':
            params.update({
            
                'quantization_config': grpc.QuantizationConfig(
                    scalar=grpc.ScalarQuantization(
                        type=grpc.QuantizationType.Int8
                   )
                )
            })

        response = await self.client.async_grpc_collections.Create(
            grpc.CreateCollection(**params)
        )
            
        return response

    async def verify_collection(self):
        try:
            response = await self.client.async_grpc_collections.Get(
                grpc.GetCollectionInfoRequest(
                    collection_name=os.getenv('COLLECTION_NAME')
                )
            )
        except grpc.RpcError as e:
            print(e)
            return False
        
        return response

    async def delete_collection(self):
        response = await self.client.async_grpc_collections.Delete(
            grpc.DeleteCollection(
                collection_name=os.getenv('COLLECTION_NAME')
            )
        )
        
        return response

    async def insert_vectors(self, data: List[dict]):
        """
        data : list of dict
        List{
            dict(
                id : int (random generated),  # not set through args
                vector : list[float]
                payload : Dict
            )
        }       
        """
        
        points = []
        for row in data:
            payload={}
            for key, value in row['payload'].items():
                payload[key] = grpc.Value(str_value=value)
            
            points.append(
            grpc.PointStruct(
                    id=grpc.PointId(generate_id()),
                    payload=payload,
                    vectors=grpc.Vectors(
                        vector=grpc.Vector(row['vector'])
                    ),
                )
            )
            if len(points) % self.config['upload_size'] == 0:
                response = await self.client.async_grpc_points.Upsert(
                    grpc.UpsertPointsRequest(
                        collection_name=os.getenv('COLLECTION_NAME'),
                        points=points,
                    )
                )
                points = []
                print(response)
            
        if len(points) % self.config['upload_size'] != 0:
            response = await self.client.async_grpc_points.Upsert(
                grpc.UpsertPointsRequest(
                    collection_name=os.getenv('COLLECTION_NAME'),
                    points=points,
                )
            )
        
        return response    
    
    async def search(self, vector: List[float]):
        response = await self.client.async_grpc_points.Search(
            grpc.SearchRequest(
                collection_name=os.getenv('COLLECTION_NAME'),
                limit=1,
                vector=vector,
                with_payload=grpc.WithPayloadSelector(enable=True),
            )
        )
        
        return response
        
if __name__ == '__main__':
    vecdb = VectorDB()
    vecdb.create_collections()