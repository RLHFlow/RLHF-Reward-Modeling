from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from generate_pipeline import ArmoRMPipeline


def main():
    # Create Reward Model Pipeline 
    prompt = 'What are some synonyms for the word "beautiful"?'
    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    # score the messages
    response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'
    score1 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])
    print(score1)
    # {'score': 0.11083984375}

    response2 = '''Certainly! Here are some synonyms for the word "beautiful":

    1. Gorgeous
    2. Lovely
    3. Stunning
    4. Attractive
    5. Pretty
    6. Elegant
    7. Exquisite
    8. Handsome
    9. Charming
    10. Alluring
    11. Radiant
    12. Magnificent
    13. Graceful
    14. Enchanting
    15. Dazzling

    These synonyms can be used in various contexts to convey the idea of beauty.'''
    score2 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}])
    print(score2)
    # {'score': 0.166015625}

    response3 = 'Sorry i cannot answer this.'
    score3 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response3}])
    print(score3)
    # {'score': 0.038330078125}

    # Observation: changing #1, #8 or #15 to Ugly, #1 leads to biggest drop in rewards, followed by #15 and then #8.
    response4 = '''Certainly! Here are some synonyms for the word "beautiful":

    1. Ugly
    2. Lovely
    3. Stunning
    4. Attractive
    5. Pretty
    6. Elegant
    7. Exquisite
    8. Handsome
    9. Charming
    10. Alluring
    11. Radiant
    12. Magnificent
    13. Graceful
    14. Enchanting
    15. Dazzling

    These synonyms can be used in various contexts to convey the idea of beauty.'''
    score4 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response4}])
    print(score4)
    # {'score': 0.11572265625}

    response4 = '''Certainly! Here are some synonyms for the word "beautiful":

    1. Gorgeous
    2. Lovely
    3. Stunning
    4. Attractive
    5. Pretty
    6. Elegant
    7. Exquisite
    8. Ugly
    9. Charming
    10. Alluring
    11. Radiant
    12. Magnificent
    13. Graceful
    14. Enchanting
    15. Dazzling

    These synonyms can be used in various contexts to convey the idea of beauty.'''
    score4 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response4}])
    print(score4)
    # {'score': 0.1279296875}

    response4 = '''Certainly! Here are some synonyms for the word "beautiful":

    1. Gorgeous
    2. Lovely
    3. Stunning
    4. Attractive
    5. Pretty
    6. Elegant
    7. Exquisite
    8. Handsome
    9. Charming
    10. Alluring
    11. Radiant
    12. Magnificent
    13. Graceful
    14. Enchanting
    15. Ugly

    These synonyms can be used in various contexts to convey the idea of beauty.'''
    score4 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response4}])
    print(score4)
    # {'score': 0.12451171875}

    response5 = '''Certainly! Here are some synonyms for the word "beautiful":

    1. Ugly
    2. Lovely
    3. Stunning
    4. Attractive
    5. Pretty

    These synonyms can be used in various contexts to convey the idea of beauty.'''
    score5 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response5}])
    print(score5)
    # {'score': 0.1015625}

    # Observation: changing the word list to Ugly still gets a higher score than "sorry I don't know".
    response6 = '''Certainly! Here are some synonyms for the word "beautiful":

    1. Ugly

    These synonyms can be used in various contexts to convey the idea of beauty.'''
    score6 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response6}])
    print(score6)
    # {'score': 0.056640625}


if __name__ == '__main__':
    main()