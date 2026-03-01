from multiprocessing.connection import Client
from dataclasses import asdict
from .base import Handler, Event

class SocketHandler(Handler):

    def __init__(self,
        host : str,
        port : int,
    ) -> None:
        super().__init__()

        self.client = Client(address=(host, port))

    def consume(self, event : Event) -> None:
        
        event_data = asdict(event)
        event_data['level'] = event.level.name

        self.client.send(event_data)