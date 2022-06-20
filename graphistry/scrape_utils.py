import hashlib
import re
import requests
import os
import json
from urllib.parse import urlparse
from collections import Counter
import pandas as pd

import tempfile
from datetime import datetime

# pip3 install newspaper3k
import newspaper

# pip install textract
import textract

from graphistry.util import setup_logger
from graphistry.constants import VERBOSE, TRACE

logger = setup_logger("scrape_utils", verbose=VERBOSE, fullpath=TRACE)

URL_REGEX = r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"

BASE_SCHEMA = {
    "article_title": "",
    "document": "",
    "summary": "",
    "keywords": [],
    "authors": [],
    "tags": [],
    "meta_title": "",
    "hashes": "",
    "urls_from_document": [],
    "url": "",
}

DATAFRAME_SCHEMA = {
    "article_title": "",
    "document": "",
    "summary": "",
    "meta_title": "",
    "hashes": "",
    "url": "",
}

EMPTY_DATAFRAME = pd.DataFrame(columns=BASE_SCHEMA.keys())


headers = requests.utils.default_headers()
headers.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
    }
)


def hash_document(document):
    document = document.encode("utf-8")

    md5Hashed = hashlib.md5(document).hexdigest()
    # sha1Hashed = hashlib.sha1(document).hexdigest()
    # res = {"MD5": md5Hashed}

    return md5Hashed


def extract_all_urls(document):
    url = re.findall(URL_REGEX, document)
    return url


def get_article_by_url(url):
    article = newspaper.Article(url)
    article.download()
    if not article.download_state:
        logger.info("Article {} could not be downloaded".format(url))
        logger.info(article.download_exception_msg)
        return None
    try:
        article.parse()
    except newspaper.ArticleException:
        logger.info(newspaper.ArticleException)
        return None
    return article


def get_article_information(url):
	"""Get article information from a url

	Args:
	url (_type_): url

	Returns:
	_type_: dict of article information
	"""
	# main scraper function
	import warnings

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		article = get_article_by_url(url)
		if article is not None:
			article.nlp()
			logger.info("Returning Dictionary of Scraped Article information")
			return {
				"article_title": article.title,
				"document": article.text,
				"summary": article.summary,
				"keywords": article.keywords,
				"authors": article.authors,
				"tags": list(article.tags),
				"meta_title": article.meta_description,
				"hashes": hash_document(article.text),
				"urls_from_document": extract_all_urls(article.text),
				"url": url,
			}
		else:
			logger.info("* NULL *" * 4)
			logger.info("Nothing to Return from Article")
			return BASE_SCHEMA


def get_pdf_from_url(url_to_pdf):
    res = requests.get(url_to_pdf)

    temp = tempfile.NamedTemporaryFile(suffix=".pdf")
    text = None
    try:
        temp.write(res.content)
        temp.seek(0)  # after writing you have to return it to beginning of text
        text = textract.process(temp.name, encoding="utf-8")  # , method='pdfminer')
        text = text.decode("utf-8", errors="ignore")
    finally:
        temp.close()

    if text:
        doc_urls = extract_all_urls(text)
        res = BASE_SCHEMA.copy()
        res["url"] = url_to_pdf
        res["document"] = text
        res["urls_from_document"] = doc_urls
        res["hashes"] = hash_document(text)
        return res
    return None


def is_pdf(url):
    if isinstance(url, dict):
        return False
    if url.endswith(".pdf"):
        return True
    return False


def is_resource(resource, document_resource):
    if isinstance(resource, dict):
        for k in document_resource:
            assert k in resource
        return True
    return False


def is_url(url_or_document):
    try:
        p = urlparse(url_or_document)
    except:
        return False
    if p.scheme != "" and p.netloc != "":
        return True
    return False

def make_metadata_file(path_to_file):
    import os
    pathfile, ext = os.path.splitext(path_to_file)
    pathfile += "_metadata"+ext
    return pathfile
    


class Scraper:
	"""
	Takes in a url (and url to pdf) or document, if url, scrapes it, and returns a DataFrame of the scraped information.
	"""

	# columns = ['url','article_title', 'document', 'summary', 'keywords', 'authors', 'tags', 'meta_title', 'hashes', 'urls_from_document', 'resource', 'predictions', 'MONEY', 'PERSON', 'FAC', 'ORG', 'PRODUCT', 'DATE', 'EVENT', 'info', 'create_date']

	def __init__(self, tracker_file, fresh_scrape=True, *args, **kwargs):
		self.tracker_file = tracker_file
		self.tracker_file_metadata = make_metadata_file(tracker_file)
		self.fresh = fresh_scrape
		self.url = None
		self.res = None
		self.ext = None
		self.info = None
		self.preds = None
		self.hash = hash_document
		self._document_resource_keys = list(BASE_SCHEMA.keys())
		self._get_metadata()
		self.df = self.get_all(verbose=False)

	def __call__(self, url_or_document_or_resource_or_pdf, *args, **kwargs):
		try:
			url_or_document_or_resource_or_pdf = url_or_document_or_resource_or_pdf.strip()
			if self.hash(url_or_document_or_resource_or_pdf) in self.hashes:
				logger.info("Already ingested, returning resource")
				return self.find_resource(url_or_document_or_resource_or_pdf)

			self.time_created = datetime.now()
			logger.info("-" * 60)
			if self.is_url(url_or_document_or_resource_or_pdf) and not self.is_pdf(
				url_or_document_or_resource_or_pdf
			):
				# this will scrape the url
				self._url_to_resource(url_or_document_or_resource_or_pdf)
			else:  # will check if it is a pdf or document
				self._extract_resource(url_or_document_or_resource_or_pdf)
			return self.res
		except Exception as e:
			logger.info(e)
			return EMPTY_DATAFRAME

	def is_url(self, url_or_document):
		return is_url(url_or_document)

	def is_pdf(self, url_or_document):
		return is_pdf(url_or_document)

	def is_resource(self, resource):
		return is_resource(resource, self._document_resource_keys)

	def is_image(self, image_url):
		# todo
		pass
	
	def _get_metadata(self):
		self.hashes = []
		self.url_counts = Counter()
		if not self.fresh:
			metadata = self._get_session_metadata()
			if metadata:
				self.hashes = metadata["hashes"]
				self.url_counts = metadata["url_counts"]

	def get(self):
		return self.res

	def _to_dataframe(self, res):
		# res = {k:v for k, v in res.items() if k in DATAFRAME_SCHEMA}
		return pd.DataFrame([res])

	def get_all(self, verbose=True):
		if os.path.exists(self.tracker_file):
			df = pd.read_csv(self.tracker_file)
			self.df = df
			return df
		logger.info(f"No Tracker Data found, returning Empty DataFrame") if verbose else None
		return EMPTY_DATAFRAME

	def find_resource(self, url_or_document_or_resource_or_pdf):
		self.get_all()
		df = self.df
		if url_or_document_or_resource_or_pdf in df["url"].values:
			return df[df["url"] == url_or_document_or_resource_or_pdf]
		elif url_or_document_or_resource_or_pdf in df["document"].values:
			return df[df["document"] == url_or_document_or_resource_or_pdf]
		else:
			logger.info(f"{url_or_document_or_resource_or_pdf} not found in Tracker")
			return EMPTY_DATAFRAME

	def _url_to_resource(self, url):
		resource = get_article_information(url)
		if resource:
			self.url = url
			self.hashes.append(self.hash(url))
			self.url_counts.update(
				[url]
			)  # how many times do you visit the url (if you care about changing resources, for example)
			self._extract_resource(resource)

	def _document_to_resource(self, document):
		# takes care if the data is just plain text that should be parsed downstream
		self.url = None
		self.hashes.append(self.hash(document))
		doc_urls = extract_all_urls(document)
		resource = BASE_SCHEMA.copy()
		resource["document"] = document
		resource["urls_from_document"] = doc_urls
		resource["hashes"] = hash_document(document)
		logger.info("Resource is a document")
		return resource

	def _pdf_to_resource(self, url_to_pdf):
		resource = get_pdf_from_url(url_to_pdf)
		if resource:
			self.url = url_to_pdf
			self.hashes.append(self.hash(url_to_pdf))
			self.url_counts.update(
				[url_to_pdf]
			)  # how many times do you visit the url (if you care about changing resources, for example)
			logger.info("Scraped PDF")
			return resource

	def _extract_resource(self, resource):
		if resource is None:
			logger.info("Resource is None")
			return
		if self.is_pdf(resource):
			resource = self._pdf_to_resource(resource)
		elif not self.is_resource(resource):
			resource = self._document_to_resource(resource)
		self.res = self._to_dataframe(resource)  # convert to dataframe
		self._add_to_store()
		self._add_session_metadata()

	def _add_to_store(self):
		if self.tracker_file:
			with open(self.tracker_file, "a") as f:
				if f.tell() == 0:
					header = BASE_SCHEMA.keys()
				else:
					header = False
				self.res.to_csv(f, header=header, index=False)
			if self.url is not None:
				logger.info(f"Added [ {self.url} ] to `{self.tracker_file}`")
			else:
				logger.info(f"Added [[ {self.res['document'].values[0][:60]}..]] to `{self.tracker_file}`")
		else:
			logger.info("No tracker file specified")

	def _add_session_metadata(self):
		# get previous session metadata if any, and add to it
		metadata = self._get_session_metadata()
		if isinstance(metadata['url_counts'], dict): # since it is a dict, you need to convert it to a Counter
			metadata['url_counts'] = Counter(metadata['url_counts'])
		metadata['url_counts'].update(self.url_counts)
		metadata['hashes'].extend(self.hashes)
		metadata['hashes'] = list(set(metadata['hashes']))
		with open(self.tracker_file_metadata, "w") as f:
			json.dump(metadata, f)
	
	def _get_session_metadata(self):
		if os.path.exists(self.tracker_file_metadata):
			with open(self.tracker_file_metadata, "r") as f:
				metadata = json.load(f)
			return metadata
		return {'hashes':[], 'url_counts': Counter()}


def ParallelScraper(urls, tracker_file, n_jobs=-1, *args, **kwargs):
    """
    Takes in a list of urls, and returns a dataframe of the scraped information using joblib.Parallel.
    """
    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib

    scrape = Scraper(tracker_file)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with tqdm_joblib(desc="scraping", total=len(urls)) as progress_bar:
            Parallel(n_jobs=n_jobs)(delayed(scrape)(x) for x in urls)

    return scrape.get_all()

