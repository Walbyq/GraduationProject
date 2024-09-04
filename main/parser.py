import concurrent.futures  # Библиотека для потоков
import os  # Библиотека для работы с файлами
import random  # Библиотека для генерации рандомных чисел
import time  # Библиотека для работы с задержками
import requests  # Библиотека для отправки запросов к сайту
from bs4 import BeautifulSoup  # Библиотека для парсинга

# Библиотека для веб-драйвера
from selenium import webdriver
from selenium.common import TimeoutException, ElementNotInteractableException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


# Парсим изображения с помощью Selenium
def parse_with_selenium(query):
    query = query.replace(' ', '+')
    length = 0
    srcs = []
    browsers = [
        {
            'name': 'Chrome',
            'url': f"https://google.com/search?tbm=isch&q={query}",
            'img_class': 'YQ4gaf',
            'button_selector': '.LZ4I'
        },
        {
            'name': 'Yandex',
            'url': f"https://yandex.ru/images/search?text={query}",
            'img_class': 'ContentImage-Image ContentImage-Image_clickable',
            'button_selector': '.Button2.Button2_width_max.Button2_size_l.Button2_view_action.FetchListButton-Button'
        }
    ]

    for browser in browsers:
        driver = webdriver.Chrome()
        driver.get(browser['url'])

        while True:
            last_count = length
            length = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"
                                           "var lenOfPage=document.body.scrollHeight;"
                                           "return lenOfPage;")
            time.sleep(0.8)
            if last_count == length:
                try:
                    button = WebDriverWait(driver, 0.5).until(
                        ec.presence_of_element_located((By.CSS_SELECTOR, browser['button_selector']))
                    )
                    button.click()
                except TimeoutException:
                    print(f"{browser['name']} scrolling is complete.")
                    break
                except ElementNotInteractableException:
                    print(f"{browser['name']} scrolling is complete.")
                    break

        # Получаем HTML и преобразовываем его в объект BeautifulSoup
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Вытаскиваем все теги <img>
        tags = soup.find_all("img", {'class': browser['img_class']})
        tags = [tag for tag in tags if ' '.join(tag.get('class')) == browser['img_class']]

        # Из тегов вытаскиваем атрибуты "src", пропуская формат изображений base64
        srcs.extend([tag.get('src') for tag in tags if not tag.get('src').startswith('data:image')])

    # Убираем дублирующиеся ссылки
    unique_srcs = list(set(srcs))
    random.shuffle(srcs)

    print(f"Images found: {len(unique_srcs)}")
    return unique_srcs


# Транслитерация и замена символов
def transliterate(query):
    rus_to_eng = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '',
        'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya', ' ': '_'
    }
    title = ''.join(rus_to_eng.get(c, c) for c in query.lower())
    return title


# Загрузка изображений локально, используя потоки
def streaming_download(path, title, srcs, numbers=None):
    def download_image(url, i):
        response = requests.get(url)
        with open(f'{path}/{title}{i}.jpg', 'wb') as f:
            f.write(response.content)

    if not os.path.exists(path):
        os.makedirs(path)

    if numbers is None:
        numbers = range(len(srcs))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for number, src in zip(numbers, srcs):
            executor.submit(download_image, src, number)
    return
