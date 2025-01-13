import requests

# 设置目标 URL
url = "https://www.archdaily.cn/search/api/v1/cn/all?q=公寓 住宅&page=9"

# 设置请求头（包括 cookies 和其他必要的 headers）
headers = {

    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,fr-FR;q=0.5,fr;q=0.4",
    "cache-control": "max-age=0",
    "cookie": "__io_first_source=archdaily.com; __io=bfa4aefcd.a9679544d_1723439136510; _ga=GA1.2.1792400690.1723439137; ad-consented_at=2029-08-11T05:06:09.550Z; _hjSessionUser_270045=eyJpZCI6IjBiMWM5NjE4LTc5M2YtNTcxZi1hMDY5LTg5NDc4ZTA1YzYwZCIsImNyZWF0ZWQiOjE3MjM0MzkxNDUwODMsImV4aXN0aW5nIjp0cnVlfQ==; __io_unique_25768=9; __io_unique_34359=9; __io_r=bing.com; __io_pr_utm_campaign=%7B%22referrerHostname%22%3A%22www.bing.com%22%7D; __gads=ID=5888461c79769fb0:T=1723439136:RT=1731143645:S=ALNI_Mbt1NIBZCwisB4FvC-CillSFqpRLg; __gpi=UID=00000ebd661aae6b:T=1723439136:RT=1731143645:S=ALNI_MY-HtDT9y44CiBojmBnbSYu7Xsxfg; __eoi=ID=ec2a01e8488b2f16:T=1723439136:RT=1731143645:S=AA-AfjYSjk7UVzGFyhykiVY7FJ0h; FCNEC=%5B%5B%22AKsRol-prfgeXFJJlqTU8OsEV0UKGDsX7Ltk1qlVZJtOJTuuPcg0IYlWJ7YTt6zP9-Ajv06Bu9ZXinYmAqkIesCi_nmRcXf-M-819CLyIjAHu3ofNue3VIpS1UUteNwS9LjL5Xr-qlpidAaSr5ekSyTz1QaJH_qs2A%3D%3D%22%5D%5D; __io_lv=1731156554541; _gid=GA1.2.220344181.1731504555; _hjSession_270045=eyJpZCI6IjFlNTcxODE2LTNmZDItNDFiNy04ZDFjLWI2ZDA5ZThjNzdlYyIsImMiOjE3MzE1MDQ1NTYwMzUsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjowLCJzcCI6MH0=; _ga_MZ4C2V2JVK=GS1.2.1731507433.15.1.1731508525.0.0.0",
    "if-none-match": 'W/"a3f781e761ba3baf847b81f649e5d340"',
    "priority": "u=0, i",
    "sec-ch-ua": '"Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
}

# 发送请求
response = requests.get(url, headers=headers)

# 打印页面内容（或解析页面）
print(response.text["results"])  # 打印 HTML 内容