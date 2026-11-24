import asyncio
from fastapi import Depends
from neo4j import GraphDatabase, Driver, Session
from src.envirenment import env, Neo4jConfig
from typing import Generator
from functools import lru_cache

@lru_cache
def get_config() -> Neo4jConfig:
    return env.neo4j['local']

_driver : Driver = get_config().driver

async def init_driver(config : Neo4jConfig):

    global _driver

    for attempt in range(10):

        try:
            print(f"Attempt {attempt}", flush=True)

            _driver = GraphDatabase.driver(
                config.uri,
                auth=(config.user, config.password),
            )

            _driver.verify_connectivity()

            print("Connected to database")
            return

        except Exception as e:
            print(f"Neo4j connection failed: {repr(e)}")
            await asyncio.sleep(5)

    raise RuntimeError("Could not connect to Neo4j")

def close_driver() -> None:
    """Close the module-level driver (called at shutdown)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def get_driver() -> Driver:
    """Return the live driver; raises if not initialised."""
    if _driver is None:
        raise RuntimeError("Neo4j driver has not been initialised.")
    return _driver


def get_session(
    driver: Driver = Depends(get_driver),
    settings: Neo4jConfig = Depends(get_config),
) -> Generator[Session, None, None]:
    """FastAPI dependency that yields a Neo4j session."""
    with driver.session(database=settings.neo4j_database) as session:
        yield session