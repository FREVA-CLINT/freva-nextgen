import json
from datetime import datetime, timedelta
import random
import uuid
import os
from pypgstac.db import PgstacDB
from pypgstac.load import Loader

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers as opensearch_helpers


NUM_ITEMS = 100000 
NUM_COLLECTIONS = 5

def generate_test_data():
    """Generate synthetic STAC test metadata."""
    collections = {}
    items = []
    project = ["CMIP6", "CORDEX", "CMIP5", "CMIP7", "CORDEX-Adjust"]

    for i in range(NUM_COLLECTIONS):
        collection_id = project[i]
        start_year = random.randint(1980, 2020)
        end_year = start_year + random.randint(1, 10)
        collections[collection_id] = {
            "id": collection_id.lower(),
            "stac_version": "1.0.0",
            "type": "Collection",
            "description": f"Test collection {i+1} for performance testing",
            "license": "MIT",
            "extent": {
                "spatial": {
                    "bbox": [[-180, -90, 180, 90]]
                },
                "temporal": {
                    "interval": [[
                        f"{start_year}-01-01T00:00:00Z", 
                        f"{end_year}-12-31T23:59:59Z"
                    ]]
                }
            },
            "providers": [{
                "name": f"Provider {i+1}",
                "roles": ["producer", "processor"],
                "url": f"https://www.freva.dkez.de"
            }]
        }

    collection_counters = {coll: 0 for coll in project}

    variables = ["temperature", "precipitation", "wind_speed", "humidity", "pressure"]
    models = ["cesm2", "mpi-esm1-2-hr", "mpi-esm1-2-lr", 
              "miroc-miroc6-clmcom-kit-cclm-6-0-clm2-v1", 
              "ec-earth-consortium-ec-earth3-veg-clmcom-btu-icon-2-6-5-rc-nukleus-x2yn2-v1"]
    experiments = ["historical", "rcp45", "rcp85", "ssp126", "ssp585"]
    frequencies = ["daily", "monthly", "hourly", "yearly", "seasonal"]
    
    for i in range(NUM_ITEMS):
        collection_id = project[i % NUM_COLLECTIONS]
        collection = collections[collection_id]
        count = collection_counters[collection_id]
        item_id = f"item-{collection_id.lower()}-{count}"
        collection_counters[collection_id] += 1
        start_date_str = collection["extent"]["temporal"]["interval"][0][0]
        end_date_str = collection["extent"]["temporal"]["interval"][0][1]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%M:%SZ")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%dT%H:%M:%SZ")
        item_start = start_date + timedelta(days=random.randint(0, (end_date - start_date).days - 30))
        item_end = item_start + timedelta(days=random.randint(1, 30))
        item_start_str = item_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        item_end_str = item_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        institute = f"institute-{random.randint(1, 10)}"
        model = random.choice(models)
        experiment = random.choice(experiments)
        realization = f"r{random.randint(1, 5)}i{random.randint(1, 3)}p{random.randint(1, 3)}f{random.randint(1, 2)}"
        frequency = random.choice(frequencies)
        variable = random.choice(variables)
        version = f"v{random.randint(2018, 2024)}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
        href_path = f"/work/{institute}/{model}/{experiment}/{realization}/{frequency}/{variable}/{version}/{item_id}.nc"
        item = {
            "id": item_id,
            "type": "Feature",
            "stac_version": "1.0.0",
            "collection": collection_id.lower(),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]
                ]]
            },
            "properties": {
                "datetime": item_start_str,
                "start_datetime": item_start_str,
                "end_datetime": item_end_str,
                "institute": institute,
                "model": model,
                "experiment": experiment,
                "realization": realization,
                "frequency": frequency,
                "variable": variable,
                "version": version
            },
            "assets": {
                "data": {
                    "href": href_path,
                    "type": "application/x-netcdf",
                    "roles": ["data"]
                }
            }
        }
        items.append(item)
        
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} items...")

    return {
        "collections": collections,
        "items": items
    }


class PgstacIngestor:
    def __init__(self):
        self.db_config = {
            "POSTGRES_USER": os.getenv("POSTGRES_USER", "pgstac"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "secret"),
            "PGHOST": os.getenv("PGHOST", "localhost"),
            "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432")
        }

    def create_dsn(self):
        """Create database connection string"""
        return f"postgresql://{self.db_config['POSTGRES_USER']}:{self.db_config['POSTGRES_PASSWORD']}@{self.db_config['PGHOST']}:{self.db_config['POSTGRES_PORT']}/postgis"

    def ingest_data(self, stac_data):
        """Ingest collections and items into pgSTAC"""
        try:
            db = PgstacDB(dsn=self.create_dsn(), debug=True)
            loader = Loader(db)
            collections = list(stac_data["collections"].values())
            loader.load_collections(collections)

            batch_size = 1000
            items = stac_data["items"]
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                loader.load_items(batch)
                print(f"Loaded items {i+1} to {min(i+batch_size, len(items))}")
            return True

        except Exception as e:
            return False

class ElasticIngestor:
    def __init__(self):
        scheme = "https" if os.getenv("ES_USE_SSL", "false").lower() == "true" else "http"
        host = os.getenv("ES_HOST", "localhost")
        port = os.getenv("ES_PORT", "9203")
        
        self.es_config = {
            "hosts": [{'host': host, 'port': int(port)}],
            "use_ssl": scheme == "https",
            "verify_certs": os.getenv("ES_VERIFY_CERTS", "false").lower() == "true",
            "ssl_show_warn": False,
            "headers": {"Content-Type": "application/json"},
            "connection_class": RequestsHttpConnection,
            "compatibility_mode": True
        }
        self.collections_index = os.getenv("STAC_COLLECTIONS_INDEX", "collections")
        self.items_index_prefix = os.getenv("STAC_ITEMS_INDEX_PREFIX", "items")

    def create_indices(self, client, collection_ids):
        """Create or update necessary indices with mappings"""
        base_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "collection": {"type": "keyword"},
                    "properties": {
                        "properties": {
                            "experiment": {"type": "keyword"},
                            "institute": {"type": "keyword"},
                            "model": {"type": "keyword"},
                            "variable": {"type": "keyword"},
                            "frequency": {"type": "keyword"},
                            "start_datetime": {"type": "date"},
                            "end_datetime": {"type": "date"},
                            "datetime": {"type": "date"}
                        }
                    },
                    "assets": {
                        "properties": {
                            "data": {
                                "properties": {
                                    "href": {"type": "keyword"}
                                }
                            }
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        if not client.indices.exists(index=self.collections_index):
            print(f"Creating index: {self.collections_index}")
            client.indices.create(index=self.collections_index, body=base_mapping)
        for collection_id in collection_ids:
            items_index = f"{self.items_index_prefix}_{collection_id.lower()}"
            if not client.indices.exists(index=items_index):
                print(f"Creating index: {items_index}")
                client.indices.create(index=items_index, body=base_mapping)

    def ingest_data(self, stac_data):
        """Ingest collections and items into OpenSearch"""

            
        try:
            client = OpenSearch(**self.es_config)
            self.create_indices(client, stac_data["collections"].keys())
            collections_actions = [
                {
                    "_index": self.collections_index,
                    "_id": collection["id"],
                    "_source": collection
                }
                for collection in stac_data["collections"].values()
            ]

            opensearch_helpers.bulk(client, collections_actions)
            print(f"Indexed {len(collections_actions)} collections")
            batch_size = 1000
            items = stac_data["items"]
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                
                batch_actions = [
                    {
                        "_index": f"{self.items_index_prefix}_{item['collection']}",
                        "_id": item["id"],
                        "_source": item
                    }
                    for item in batch
                ]
                
                opensearch_helpers.bulk(client, batch_actions)
                print(f"Indexed items {i+1} to {min(i+batch_size, len(items))}")
            return True

        except Exception as e:
            return False

def main():
    stac_data = generate_test_data()
    print("========== PostgreSQL PgSTAC Ingestion Test ==========")
    import time
    pg_start_time = time.time()
    
    pg_ingestor = PgstacIngestor()
    pg_success = pg_ingestor.ingest_data(stac_data)
    
    pg_end_time = time.time()
    pg_duration = pg_end_time - pg_start_time
    
    if pg_success:
        print(f"PostgreSQL ingestion completed in {pg_duration:.2f} seconds")
    

    print("========== OpenSearch Ingestion Test ==========")
    os_start_time = time.time()
    
    os_ingestor = ElasticIngestor()
    os_success = os_ingestor.ingest_data(stac_data)
    
    os_end_time = time.time()
    os_duration = os_end_time - os_start_time
    
    if os_success:
        print(f"OpenSearch ingestion completed in {os_duration:.2f} seconds")

if __name__ == "__main__":
    main()

