import codecs
from bs4 import BeautifulSoup
import json
import yaml
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from collections import defaultdict
import xml.etree.ElementTree as ET
import nltk.data
import re
import ast
import string


EMPTY_TOKEN = '__empty__'

# Global variables
sentence_tokenizer = None
stops = None

# Evaluation scripts from TREC:
# http://trec.nist.gov/data/web/12/gdeval.pl
gdeval = 'eval/gdeval.pl'


def read_file(file_path, mode='r', encoding="utf-8"):
    """Generic function to read from a file"""
    with codecs.open(file_path, mode, encoding=encoding) as fp:
        return fp.read().strip()


def read_qrels(file_path, mode='r', encoding='utf-8'):
    with codecs.open(file_path, mode, encoding=encoding) as fp:
        lines = [line.strip().split() for line in fp]
    return lines


def read_query(query_files):
    # Reads XML query files
    query_id2text = defaultdict(dict)
    for query_file in query_files:
        tree = ET.parse(query_file)
        root = tree.getroot()
        for query in root.findall('topic'):
            qid = int(query.attrib['number'])
            for attr in query:
                if attr.tag in ['query', 'description']:
                    query_id2text[qid][attr.tag] = attr.text.strip()
    return query_id2text


def write_file(content, file_path, mode='w', encoding='utf-8'):
    """Generic function to write to a file"""
    with codecs.open(file_path, mode, encoding=encoding) as fid:
        fid.write(content)


def split_in_sentences(texts):
    if sentence_tokenizer is None:
        global sentence_tokenizer
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return list(map(sentence_tokenizer.tokenize, texts))


def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


def preprocess_text(text, tokenize=False, ner=False, stem=False, stopw=False, all_lower=False, strip_punct=True):
    """
    Preprocesses and cleans text
    :param ner: Do Named Entity Recognition and join into one word
    :param stem: Stem text
    :param stopw: Remove stopwords
    :param all_lower: lowercase text
    :param strip_punct: strips punctuation
    :return: preprocessed text
    """

    # Clean the text
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"i\.e\.", "", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r'"', " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r"^e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"^b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"^u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    if ner:
        tokenized_text = word_tokenize(text)
        tagged_text = pos_tag(tokenized_text)
        chunked_text = ne_chunk(tagged_text, binary=True)

        named_entities = extract_entity_names(chunked_text)
        for named_entity in named_entities:
            entity = named_entity.replace(".", "")
            entity = re.sub(r'\s+', "_", entity)
            text = text.replace(named_entity, entity)

    if all_lower:
        text = text.lower()

    if stopw:
        global stops
        if stops is None:
            try:
                stops = set(stopwords.words("english"))
            except Exception as e:
                print("%s - Please download english stopwords from NLTK" % e)
                exit()
        text = [word.strip() for word in text.split() if word not in stops]
        text = " ".join(text)

    if tokenize:
        text = word_tokenize(text)
        text = " ".join(text)

    # shorten words to their stems
    if stem:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    if strip_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))

    text = text.strip()

    # Empty string
    if text == '':
        return EMPTY_TOKEN

    return text


def read_topics_from_xml(file, topics_dict):

    """
    Adds/overwrites to topics_dict new topic numbers as index and query and subtopics as value
    :param file: XML file containing TREC topics
    :param topics_dict: defaultdict(dict) variable
    :return: topics_dict updated
    """

    # Read XML file
    xml = read_file(file)

    # Pass to HTML
    html = BeautifulSoup(xml, "html.parser")

    # Get topics
    topics = html.findAll('topic')
    for topic in topics:
        # Get topic number + query and description
        topic_number = int(topic['number'])
        query = [topic.query.getText().strip(),
                 topic.description.getText().strip()]
        topics_dict[topic_number]['query'] = query

        # Get subtopics
        subtopics = topic.findAll('subtopic')
        topics_dict[topic_number]['subtopics'] = [x.getText().strip() for x in subtopics]

    return topics_dict


def parse_config(config_path):
    file_type = config_path.split('.')[-1]
    if file_type == 'json':
        with open(config_path) as fid:
            config = json.loads(fid.read())
    elif file_type == 'yml' or file_type == 'yaml':
        with open(config_path, 'r') as fid:
            config = yaml.load(fid)
    else:
        raise Exception("Invalid config file (has to be yaml or json)")
    return config


def edit_config(config_dict, overload_list):

    for overload_str in overload_list:
        assert len(overload_str.split('=')) == 2, "overload expects one or more inputs of format field=value"

    keys = list(map(lambda x: x.split('=')[0], overload_list))
    values = list(map(lambda x: x.split('=')[1], overload_list))

    if isinstance(keys, list) and isinstance(values, list):
        assert len(keys) == len(values), \
            "Keys and Values should have the same number of elements"
    else:
        keys = [keys]
        values = [values]

    for key, value in zip(keys, values):
        sub_keys = key.split('.')
        entry = config_dict
        for k in sub_keys[:-1]:
            entry = entry[k]

        assert not isinstance(entry[sub_keys[-1]], dict), \
            "Cannot set non-leaf fields"

        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass

        entry[sub_keys[-1]] = value

    return config_dict
