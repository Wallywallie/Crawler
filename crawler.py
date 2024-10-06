import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_search_results(query, num_pages=1):
    results = []
    for page in range(num_pages):
        url = f"https://www.baidu.com/s?wd={query}&pn={page*10}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for result in soup.find_all('h3', class_='t'):
            title = result.get_text()
            link = result.find('a')['href']
            results.append((title, link))
    return results

def extract_info(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
    driver.get(url)
    time.sleep(2)  # 等待页面加载
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    content = soup.get_text()
    driver.quit()
    return content

def main():
    query = "彼得·卒姆托 建筑作品"
    num_pages = 3  # 爬取前3页的搜索结果
    
    results = get_search_results(query, num_pages)
    
    for title, link in results:
        print(f"Title: {title}\nLink: {link}")
        info = extract_info(link)
        print(f"Content: {info[:500]}...")  # 打印部分内容
        print("\n")

if __name__ == "__main__":
    main()