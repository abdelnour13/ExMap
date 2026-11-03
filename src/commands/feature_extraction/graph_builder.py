import torch
from src.shared.command_tools import BaseCommand
from src.shared.file_tools import FileReader
from src.commands.feature_extraction.graph_utils import Graph, EdgeTypes, NodeTypes
from src.shared.env import env

def _slugify(s : str) -> str:
    return s.lower().replace(' ','_')

def _pairs(items : list):

    v = []

    for item1, item2 in zip(items[:-1], items[1:]):
        yield '.'.join(v+[item1]), '.'.join(v+[item1,item2])
        v.append(item1)

class GraphBuilder:

    def __init__(self, 
        _host : BaseCommand,
        reader : FileReader
    ) -> None:
        self._host = _host
        self.reader = reader

    def build(self) -> Graph:

        self._host.send_info_event(
            progress=0.0,
            task="graph-construction",
            message="[STAGE 1] Graph construction..."
        )

        graph = Graph()

        for records, progress in self.reader.batched(4096):

            for record in records:
                self.update(record, graph)

            self._host.send_info_event(
                progress=progress,
                task="graph-construction",
                message=f"Processed {len(records)} records."
            )

        for domain in env.fields['industry_of_future'].domains:
            
            for d in map(str.strip, domain.domain.split(',')):

                domain_id = _slugify(d)
                subdomain_id = _slugify(domain.sub_domain)

                graph.add_node(
                    type='Domain',
                    id=domain_id,
                    properties={
                        "name" : d
                    }
                )

                graph.add_node(
                    type='SubDomain',
                    id=subdomain_id,
                    properties={
                        "name" : domain.sub_domain,
                        "definition" : domain.definition,
                    }
                )

                for alt_label in ([domain.sub_domain] + (domain.alt_names or [])):

                    alt_label_id = _slugify(alt_label)

                    graph.add_node(
                        type='AltLabel',
                        id=alt_label_id,
                        properties={
                            "name" : alt_label,
                        }
                    )

                    graph.add_edge(
                        type=EdgeTypes.SubDomain_HASALTLABEL_Domain,
                        src=subdomain_id,
                        dst=alt_label_id,
                    )

                graph.add_edge(
                    type=EdgeTypes.SubDomain_ISSUBDOMAINOF_Domain,
                    src=subdomain_id,
                    dst=domain_id
                )

        return graph

    def update(self, doc: dict, graph: Graph) -> None:

        article = doc['sentence']['document'] if doc['sentence'] is not None else None

        if article is None:
            return

        # Article
        added = graph.add_node(
            type="Article",
            id=article['id'],
            properties={
                "id" : article['id'],
                "title" : article['title'],
                "abstract" : article['abstract'],
                "date" : article['date']
            }
        )            

        # Article Labels
        if added:
            
            for method, items in article['pseudo_labels'].items():

                t = torch.tensor([i['score'] for i in items])
                
                if method not in graph._nodes['Article']._features:
                    graph._nodes['Article']._features[method] = t[None,:]
                else:
                    graph._nodes['Article']._features[method] = torch.cat([
                        graph._nodes['Article']._features[method],
                        t[None,:]
                    ], dim=0)

        # Predicate
        predicate_id = _slugify(doc['predicate'])

        graph.add_node(
            type=NodeTypes.Predicate,
            id=predicate_id,
            properties={
                "name" : doc['predicate']
            }
        )

        # Triple
        triple_id = _slugify('_'.join([doc['head'], doc['predicate'], doc['tail']]))
        head_id = _slugify(doc['head'])
        tail_id = _slugify(doc['tail'])

        graph.add_node(
            type=NodeTypes.Triple,
            id=triple_id
        )

        graph.add_edge(
            type=EdgeTypes.Triple_HASHEAD_Entity,
            src=triple_id,
            dst=_slugify(doc['head'])
        )

        graph.add_edge(
            type=EdgeTypes.Triple_HASTAIL_Entity,
            src=triple_id,
            dst=_slugify(doc['tail'])
        )

        graph.add_edge(
            type=EdgeTypes.Triple_HASPREDICATE_Predicate,
            src=triple_id,
            dst=predicate_id
        )

        graph.add_edge(
            type=EdgeTypes.Triple_BELONGSTO_Article,
            src=triple_id,
            dst=article['id']
        )

        # Entity
        for entity_id, entity in zip([head_id, tail_id],[doc['head'], doc['tail']]):

            graph.add_node(
                type='Entity',
                id=entity_id,
                properties={
                    "name" : entity
                }
            )

        # Lab
        for lab in article['teams']:

            lab_id = _slugify(lab)

            graph.add_node(
                type='Lab',
                id=lab_id,
                properties={
                    "name" : lab
                }
            )

            graph.add_edge(
                type=EdgeTypes.Article_BELONGSTO_Lab,
                src=article['id'],
                dst=lab_id
            )

        # Venue
        for venue in article['venues']:

            graph.add_node(
                type='Venue',
                id=venue,
                properties={
                    "name" : venue
                }
            )

            graph.add_edge(
                type=EdgeTypes.Article_BELONGSTO_Venue,
                src=article['id'],
                dst=venue
            )

            for v1, v2 in _pairs(venue.split('.')):

                graph.add_node(
                    type='Venue',
                    id=v1,
                    properties={
                        "name" : v1
                    }
                )

                graph.add_node(
                    type='Venue',
                    id=v2,
                    properties={
                        "name" : v2
                    }
                )

                graph.add_edge(
                    type=EdgeTypes.Venue_ISSUBVENUEOF_Venue,
                    src=v2,
                    dst=v1
                )

        # Author
        for author in article['authors']:

            author_id = _slugify(author)

            graph.add_node(
                type='Author',
                id=author_id,
                properties={
                    "name" : author
                }
            )

            graph.add_edge(
                type=EdgeTypes.Article_WRITTENBY_Author,
                src=article['id'],
                dst=author_id
            )

        # Keyword
        for keyword in article['keywords']:

            keyword = keyword['keyword']

            keyword_id = _slugify(keyword)

            graph.add_node(
                type='Keyword',
                id=keyword_id,
                properties={
                    "name" : keyword
                }
            )

            graph.add_edge(
                type=EdgeTypes.Article_HASKEYWORD_Keyword,
                src=article['id'],
                dst=keyword_id
            )

        # SubDomains
        for method, items in article['pseudo_labels'].items():

            for item in items:
                
                domain = item['label']
                domain_id = _slugify(domain)

                graph.add_node(
                    type='SubDomain',
                    id=domain_id,
                    properties={
                        "name" : domain
                    }
                )