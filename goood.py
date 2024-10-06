import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By
from typing import(List)
import logging
from abc import ABC, abstractmethod
import re

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("search_results.log"),
                        logging.StreamHandler()  # 同时输出到控制台
                    ])
logger = logging.getLogger(__name__)


class Content:
    """
    表示内容，可以是段落或图片。

    Attributes:
        content_type (str): 内容类型 ('paragraph' 或 'image')。
        content (str): 内容（段落文字或图片链接）。
        image_caption (str): 图片说明（如果内容类型是 'image'）。
    """    
    def __init__(self, type: str, content: str, image_caption: str = None):
        self.type = type  # 'paragraph' or 'image'
        self.content = content
        self.image_caption = image_caption

class Case:
    """
    表示一个案例，包含标题和多个内容项。

    Attributes:
        title (str): 案例的标题。
        contents (list): 案例中的内容项列表。
    """    
    def __init__(self, title: str):
        self.title = title
        self.contents = []

    def add_content(self, content: Content):
        self.contents.append(content)

    def print_case(self):
        for i in self.contents:
            print(i.content)

    def save(self, file_name:str):
        #sanitize file name
        # 移除或替换不允许的特殊字符，保留字母、数字、空格、连字符、下划线
        sanitized_name = re.sub(r'[\/:*?"<>|]', '_', file_name)
    
        # 如果文件名过长，进行截断
        max_length = 255 - len('.txt')  # 保证文件名+扩展名不超过255个字符
        if len(sanitized_name) > max_length:
            sanitized_name = sanitized_name[:max_length]    
        path = sanitized_name + '.txt'
        paragraphs = ''.join(i.content for i in self.contents)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(paragraphs)

    def save_text(self, file_name:str):
        #sanitize file name
        # 移除或替换不允许的特殊字符，保留字母、数字、空格、连字符、下划线
        sanitized_name = re.sub(r'[\/:*?"<>|]', '_', file_name)
    
        # 如果文件名过长，进行截断
        max_length = 255 - len('.txt')  # 保证文件名+扩展名不超过255个字符
        if len(sanitized_name) > max_length:
            sanitized_name = sanitized_name[:max_length]    
        path = sanitized_name + '.txt'
        paragraphs = ''.join(i.content for i in self.contents if i.type == "paragraph")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(paragraphs)  

# 定义抽象策略类
class CrawlStrategy(ABC): 
    @abstractmethod
    def get_search_results(self, query):
        pass

    @abstractmethod
    def extract_info(url):
        pass


class GdCrawlStrategy(CrawlStrategy):

    def get_search_results(driver : webdriver.Chrome,query:str, page:int = 1, wait_time = 1)->List[str]:
        """
        获取搜索结果链接列表。

        Args:
        query (str): 搜索关键词。
        page (int): 页码，默认为1。
        wait_time : 等待页面加载的时间，默认为1秒。

        Returns:
        List[str]: 包含搜索结果链接的列表。
        """
        results = []
        base_url = "https://www.gooood.cn/search/"
        url = f"{base_url}{query}/page/{page}"

        try:

            driver.get(url)
            time.sleep(wait_time)  # 等待页面加载

            links = driver.find_elements(By.CLASS_NAME, 'post-thumbnail')
            for i in links:
                a_tag = i.find_element(By.TAG_NAME, 'a')
                if a_tag is None:
                    continue
                results.append(a_tag.get_attribute('href'))
                logger.info(f"Found link: {a_tag.get_attribute('href')}")
        except Exception as e:
            logger.error(f"An error occurred in get_search_results: {e}")   
               

        logger.info(f"Total links found: {len(results)}")
        return results
    
    def extract_info(driver : webdriver.Chrome, url:str, wait_time = 1 ) -> Case:
        """
        从给定的URL提取案例信息。

        Args:
        url (str): 要提取信息的网页URL。
        wait_time : 等待页面加载的时间，默认为1秒。

        Returns:
        Case: 包含提取内容的Case对象。
        """

        try:
        
            driver.get(url)
            time.sleep(wait_time)  # 等待页面加载
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # get title | cover | content | picture | caption
            title = soup.find('title').get_text() if soup.find('title') else "No Title"
            #cover = soup.find('meta', property="og:image")['content']
            case = Case(title)

            post_content = soup.find('post-content')
            if post_content:
                p_tags = post_content.find_all('p')
                if p_tags:
                    for p in p_tags:
                        if p.get_text(strip=True):
                            case.add_content(Content('paragraph',p.get_text(strip=True) ))
                        a_tag = p.find('a')
                        if a_tag:
                            img_tag = a_tag.find('img')
                            if img_tag:
                                case.add_content(Content('image',img_tag['src']))
        except Exception as e:
            logger.error(f"An error occurred in extract_info: {e}")

        logging.info(f"Extraction completed for URL: {url}")

        return case

class BaiduCrawlStrategy(CrawlStrategy):
    def get_search_results(self, query):
        pass


class ArchDailyStrategy(CrawlStrategy):

    def get_search_results(driver : webdriver.Chrome,query:str, page:int = 1, wait_time = 1)->List[str]:
        """
        获取搜索结果链接列表。

        Args:
        query (str): 搜索关键词。
        page (int): 页码，默认为1。
        wait_time : 等待页面加载的时间，默认为1秒。

        Returns:
        List[str]: 包含搜索结果链接的列表。
        """
        results = []
        base_url = "https://www.archdaily.cn/search/cn/projects?q="
        url = f"{base_url}{query}&page={page}"
        try:

            driver.get(url)
            time.sleep(wait_time)  # 等待页面加载
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            a_tags = soup.find_all("a", class_="gridview__content")
            for i in  a_tags:
                if i is None:
                    continue
                results.append(i['href'])
                logger.info(f"Found link: {i['href']}")

        except Exception as e:
            logger.error(f"An error occurred in get_search_results: {e}")

        return results    
    

    def extract_info(driver : webdriver.Chrome, url:str, wait_time = 1 )-> Case:
        """
        从给定的URL提取案例信息。

        Args:
        url (str): 要提取信息的网页URL。
        wait_time : 等待页面加载的时间，默认为1秒。

        Returns:
        Case: 包含提取内容的Case对象。
        """
        try:
        
            driver.get(url)
            time.sleep(wait_time)  # 等待页面加载
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')


            title = soup.find('title').get_text() if soup.find('title') else "No Title"
            case = Case(title)

            
            start_content = soup.find("span", class_ = "less-txt hide")
            pre = start_content.find_next().find_previous_sibling()
            for sib in pre.next_siblings:
                if sib.name == "figure":
                    img = sib.find("a")["data-image"]
                    content = Content("image", img)
                    case.add_content(content)
                elif sib.name == "p":
                    para = sib.get_text(strip=True)
                    if para:
                        content = Content("paragraph", para)
                        case.add_content(content)

        except Exception as e:
            logger.error(f"An error occurred in extract_info: {e}")

        logging.info(f"Extraction completed for URL: {url}")

        return case 

# 爬虫类
class Crawler:

    def create_driver(self) -> webdriver.Chrome:
        """
        创建并配置一个新的 Chrome WebDriver 实例。
        
        返回:
        webdriver.Chrome: 配置好的 WebDriver 实例。
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
        return driver    


    def __init__(self, strategy: CrawlStrategy):
        self.strategy = strategy
        self.driver = self.create_driver()

    def set_strategy(self, strategy: CrawlStrategy):
        self.strategy = strategy

    def get_search_results(self, query, page = 1,wait_time=1):
  
        return self.strategy.get_search_results(self.driver,query,page,wait_time)
    
    def close(self):
        self.driver.quit()
        logger.info("Browser closed.")

    def extract_info(self, url,wait_time=1):
        return self.strategy.extract_info(self.driver, url, wait_time)    

query = "尼康直营店"
crawler = Crawler(ArchDailyStrategy)
urls = crawler.get_search_results(query)
for i in  urls:
    case = crawler.extract_info(i,wait_time=1)
    case.save_text(case.title)

crawler.set_strategy(GdCrawlStrategy)
urls = crawler.get_search_results(query)
for i in  urls:
    case = crawler.extract_info(i,wait_time=1)
    case.save_text(case.title)

crawler.close()
 




