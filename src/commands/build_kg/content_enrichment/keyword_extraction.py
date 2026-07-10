import torch
from typing import Dict, List, Tuple
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from definitions import hf_model_dir
from instance import cli
from torch.nn.utils.rnn import pad_sequence

def getkeyphrasetransformerClass():

    from keyphrasetransformer import KeyPhraseTransformer as _KeyPhraseTransformer

    class KeyPhraseTransformer(_KeyPhraseTransformer):

        def __init__(self, 
            model_type = "t5", 
            model_name = "snrspeaks/KeyPhraseTransformer",
            device : torch.types.Device = 'cpu'
        ) -> None:
            super().__init__(model_type, model_name)
            self.device = device
            self.to(device)

        def to(self, device : torch.types.Device) -> 'KeyPhraseTransformer':
            self.device = device
            self.model.to(device)
            return self
        
        def batch_predict(self, docs : List[str]):

            inputs = []

            for doc in docs:

                input_ids = self.tokenizer.encode(
                    doc, return_tensors="pt", add_special_tokens=True
                )[0,:]

                inputs.append(input_ids)

            inputs = pad_sequence(inputs, batch_first=True, padding_value=0).to(self.device)

            generated_ids_list = self.model.generate(
                input_ids=inputs,
                num_beams=2,
                max_length=512,
                repetition_penalty=2.5,
                length_penalty=1,
                early_stopping=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1,
            )

            preds = [
                [
                    self.tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for g in generated_ids[None,:].cpu()
                ]
                for generated_ids in generated_ids_list
            ]

            return preds
        
        def batch_get_key_phrases(self, docs: List[str], text_block_size: int = 64):

            results = []

            ids = []
            paragraphs = []

            for i,doc in enumerate(docs):

                paras = self.split_into_paragraphs(
                    doc=doc, max_tokens_per_para=text_block_size
                )

                for par in paras:
                    ids.append(i)
                    paragraphs.append(par)

            results = self.batch_predict(paragraphs)

            results_text_pairs = [([], doc) for doc in docs]

            for idx, results in zip(ids, results):
                results_text_pairs[idx][0].append(results)

            all_key_phrases = []

            for results,doc in results_text_pairs:
                key_phrases = self.filter_outputs(self.process_outputs(results), doc)
                all_key_phrases.append(key_phrases)

            return all_key_phrases

        def predict(self, doc: str):

            input_ids = self.tokenizer.encode(
                doc, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=2,
                max_length=512,
                repetition_penalty=2.5,
                length_penalty=1,
                early_stopping=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1,
            )

            preds = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids.cpu()
            ]

            return preds
        
    return KeyPhraseTransformer

@cli.register_stage(
    name="keywords_extraction",
    description=(
        "Extract additional keywords using 'KeyPhraseTransformer' T5 based keyword / topic extractor."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class KeywordsExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
    ) -> None:
        
        super().__init__(reader, writer)

        cls = getkeyphrasetransformerClass()
        self.key_phrase_transformer = cls(model_name=hf_model_dir("snrspeaks/KeyPhraseTransformer"), device='cuda')

    def transform_chunk(self, chunk : List[Dict]) -> Tuple[List[Dict], List[Dict], List[Exception]]:
        
        results = []

        keywords_list = self.key_phrase_transformer.batch_get_key_phrases([
            doc["abstract"]
            for doc in chunk
        ])

        for doc, keywords in zip(chunk, keywords_list):

            new_doc = {
                **doc,
                "keywords" : [
                    *
                    [
                        {
                            "keyword" : keyword,
                            "source"  : "api"
                        }
                        for keyword in doc['keywords']
                    ],
                    *
                    [
                        {
                            "keyword" : keyword,
                            "source"  : "keyphrase_transformer"
                        }
                        for keyword in keywords
                        if keyword not in doc['keywords']
                    ]
                ]
            }

            results.append(new_doc)

        return chunk, results, []
    
    def transform(self, doc : Dict) -> Dict:
        
        keywords = self.key_phrase_transformer.get_key_phrases(doc["abstract"])

        ### Here i updated the step to specify the source of
        ### the keyword
        doc["keywords"] = [
            *
            [
                {
                    "keyword" : keyword,
                    "source"  : "hal"
                }
                for keyword in doc['keywords']
            ],
            *
            [
                {
                    "keyword" : keyword,
                    "source"  : "keyphrase_transformer"
                }
                for keyword in keywords
                if keyword not in doc['keywords']
            ]
        ]

        return doc

    @classmethod
    def from_config(cls, config : StageConfig) -> Stage:

        cache = TextfileCache(checkpoint_filename=config.checkpoint_filename)
        
        return KeywordsExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
        )
    
if __name__ == '__main__':

    keywords_extractor = KeywordsExtractor(None, None)

    example1 = {"id": "1631926", "title": "Dashboard for monitoring student engagement in mind mapping activities", "abstract": "Our work aims at helping teachers to monitor learners ' engagement during mind mapping activities . The use of mind maps helps students to elaborate cognitive learning strategies like creating and organizing contents . To assess the quality of mind maps , teachers need tools to understand students ' choices and strategies when constructing teachers mind maps . Our have defined in collaboration with teachers a set of indicators based on learners ' interaction traces . a set of indicators based on learners ' interaction traces have been implemented and integrated in a dashboard dedicated to teachers . In this paper , Our introduce a first version of a dashboard dedicated to teachers : a dashboard dedicated to teachers design , implementation , and the results of a dashboard dedicated to teachers evaluation conducted during interviews with real teachers .", "keywords": ["dashboard", "mind mapping", "visualization"], "authors": ["Rubiela Carrillo", "Cl\u00e9ment Renaud", "Yannick Pri\u00e9", "\u00c9lise Lavou\u00e9"], "venues": ["info.eiah", "info.info-hc", "shs.edu"], "teams": ["LS2N-DUKE"], "date": "2017-07-03T00:00:00Z", "pseudo_labels": [{"model": "deepseek-chat", "sub_domains": {"Robotics": 3, "Processes and Assembly": 3, "Additive Manufacturing": 3, "Monitoring and Analysis": 1, "Eco-design": 3, "Reduction and Treatment of Effluents and Waste": 3, "Recycling / Reuse": 3, "Eco-responsible Processes": 3, "Decarbonization of Mobility": 3, "Decarbonization of Energy": 3, "Composites and New Materials": 3, "Industrial Digitalization": 3, "Digital Engineering of Products and Processes": 2, "Industrial Organization Performance": 3, "Preparation for New Jobs in the Future Industry": 3, "Change Management": 3, "User-Centered Design": 1, "Augmentation of Human Capabilities in Manufacturing": 3}, "domains": {"Advanced Production Technologies": 1, "Sustainable Industry": 3, "Industrial Digitalization": 2, "Industry 5.0": 1}}, {"model": "gemini-3.1-flash-lite", "sub_domains": {"Robotics": 3, "Processes and Assembly": 3, "Additive Manufacturing": 3, "Monitoring and Analysis": 2, "Eco-design": 3, "Reduction and Treatment of Effluents and Waste": 3, "Recycling / Reuse": 3, "Eco-responsible Processes": 3, "Decarbonization of Mobility": 3, "Decarbonization of Energy": 3, "Composites and New Materials": 3, "Industrial Digitalization": 3, "Digital Engineering of Products and Processes": 3, "Industrial Organization Performance": 3, "Preparation for New Jobs in the Future Industry": 2, "Change Management": 2, "User-Centered Design": 1, "Augmentation of Human Capabilities in Manufacturing": 3}, "domains": {"Advanced Production Technologies": 2, "Sustainable Industry": 3, "Industrial Digitalization": 3, "Industry 5.0": 1}}]}
    example2 = {"id": "4409450", "title": "Tracking motor activity in Virtual Reality to reveal cognitive functioning: A preliminary study", "abstract": "A challenge for cognitive research is the better understanding of how motor activity influences and is influenced by other cognitive domains. We developed a preliminary study to investigate whether tracking motor functioning in virtual reality provides useful insight on cognitive functioning. We chose the flankers task as an assessment measure and built a VR environment into which seven participants completed more than 1250 trials. In addition to classical results of the flankers task showing that incongruent stimuli induce slower responses than others., we also identified how individuals are able to correct their initially incorrect motor response. This indicator may shed new light into the functioning of cognitive control in the future. We discuss the potential offered by virtual reality technology for cognitive assessment through embodied considerations of cognition.", "keywords": [], "authors": ["Nicolas Ribeiro", "Toinon Vigier", "Yannick Pri\u00e9"], "venues": ["info.info-dl"], "teams": ["LS2N-DUKE"], "date": "2021-05-18T00:00:00Z"}

    from pprint import pprint

    pprint(keywords_extractor.key_phrase_transformer.batch_get_key_phrases([
        example1['abstract'],
        example2['abstract'],
    ]))