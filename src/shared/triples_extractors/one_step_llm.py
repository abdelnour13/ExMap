import asyncio
from .base import Triple, LLMBasedTripleExtractor
from openai import RateLimitError, APIError
from returns.result import Result
from typing import List, Union, Tuple, Optional
from returns.pipeline import is_successful

prompt = f"""
The task is to extract a set of relevant triples from abstracts of articles related to computer science in order to build a computer science knowledge graph.
Given a sentence, extract the set of relevant and significant triples that obey the following rules:

1- Each triple is composed of three parts: a subject, a predicate, and an object.

2- The subject, predicate, and object must be explicitly mentioned in the given sentence.

3- The predicate must be an English verb or a composition of two English verbs. Prepositions such as: to, of, with, on are allowed at the end of the predicate but are not considered valid predicates by themselves. The predicate should not contain words that imply ambiguity or non-reliability such as: often, may, could.

4- The subject and object must be noun phrases that do not contain adjectives or adverbs that do not contribute meaningfully to the name of the entity.  
For example:  
Given the sentence "Decision Tree is one of the most used models", the extracted triple should be:  
"(Decision Tree,is,model)"  
since the phrase "one of the most used" does not add meaningful information to the entity name or the triple.  

However, given the sentence "Probabilistic models are models", the extracted triple should be:  
"(Probabilistic model,is,model)"  
The adjective "Probabilistic" is kept because it adds meaningful context to the entity.

5- Generic entities such as: "This paper", "paper", "experiment", "researcher", "this work", "work", or any entities related to the presentation of the work (as commonly written in abstracts) should not be considered valid entities.

6- The subject or object should not contain more than one entity.  
For example: "GAN is composed of a generator and a discriminator"  
The extracted triples should be:  
"(GAN,composed of,generator),(GAN,composed of,discriminator)"  
and not:  
"(GAN,composed of,generator and discriminator)".

7- The answer must be formatted exactly as follows:  
"(subject_1,predicate_1,object_1)|(subject_2,predicate_2,object_2)"

8- No additional explanation should be provided in the answer.

9- If no triples that comply with the rules can be extracted, return an empty answer.

10- Examples:

a. "GAN is composed of two neural networks, a generator and a discriminator" ->  
"(GAN,composed of,generator)|(GAN,composed of,discriminator)|(Generator,is,neural network)|(Discriminator,is,neural network)"

b. "Scikit-learn, a Python library, is widely used in data science and machine learning." ->  
"(Scikit-learn,is,Python library)|(Scikit-learn,used in,data science)|(Scikit-learn,used in,machine learning)"

c. "Recent studies show good results" ->  
""

d. "LLMs are used for a wide variety of tasks" ->  
"(LLMs,used for,task)"

e. "DEC combines Deep Learning and traditional machine learning in a hybrid method to assign data points to clusters" ->  
"(DEC,combines,Deep Learning)|(DEC,combines,machine learning)|(DEC,assigns,clusters)"
"""


class OneStepLLMTriplesExtractor(LLMBasedTripleExtractor):

    def parse_triple(self, triple : str) -> Optional[Triple]:

        triple = triple.strip('()\n ')

        try:

            head, predicate, tail = triple.split(',')

            head = head.strip('[]\n ')
            predicate = predicate.strip('[]\n ')
            tail = tail.strip('[]\n ')

            return Triple(head,predicate,tail)

        except:
            return None

    async def extract_triples(self, text : str) -> Result[List[Triple], Union[RateLimitError,APIError]]:

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": text
            }
        ]

        response = await self.make_request(messages)

        return response \
            .map(lambda response : response.choices[0].message.content) \
            .map(lambda content : list(
                    filter(
                        lambda x : x is not None, 
                        map(self.parse_triple, content.split('|'))
                    )
                )
            )
    
    async def batch_extract_triples(self, sentences : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
        tasks = [self.extract_triples(sentence) for sentence in sentences]
        results = await asyncio.gather(*tasks)
        return results
    
    def extract(self, texts : List[str]) -> List[Result[List[Triple], Union[RateLimitError,APIError]]]:
        return asyncio.run(self.batch_extract_triples(texts))