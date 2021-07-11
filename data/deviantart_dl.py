# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:58:41 2020

@author: Kabey
"""

import re
from bs4 import BeautifulSoup as BS
import urllib.request
import requests
import os
import pandas as pd
import platform
import cfscrape
import time


def download_img(df,main_path,inf,sup):
    
    
    session = requests.Session()
    #scraper = cfscrape.create_scraper(delay=100)#delay
    
    for i in range(inf,sup):
        time.sleep(4)
        url=df['url'][i]
        
        #html_content = requests.get(url).text
        
        #html_content = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'}).text
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0'}
               
        proxies={'http':'50.207.31.221:80'}
        
        html_content = session.get(url,headers=headers,proxies=proxies,timeout=5).text
        #html_content = scraper.get(url,headers=headers,proxies=proxies).text
        
        soup = BS(html_content,'html.parser')
        
        '''
        with open(url) as html:
            soup = BS(html,'html.parser')
        '''
        
        #soup = BS(main_path+'sonic1.html', 'html.parser')
        
        #print(soup.prettify())
        
        vid = soup.findAll('video')
        img = soup.findAll('img')
        
        #print("img: ", img)
        temp = soup.title
        
        if str(temp).rfind("404") != -1:
            if platform.system()=="Windows":
                name = 'img\\FanartSonic_' + str(i)+ '.png'
            elif platform.system()=="Linux":
                name = 'img/FanartSonic_' + str(i)+ '.png'
            urllib.request.urlretrieve('https://worldofvoz.files.wordpress.com/2020/01/http-error-404-not-found.png', main_path+name)
        
        else:

            beginning = str(temp).find(">")+1
            ending = str(temp).rfind("<")
            title = str(temp)[beginning:ending]
            ending2 = str(title).rfind(" by")
            
            title = str(title)[0:ending2]
            
            sauce = soup.find(alt=title)
            
            if sauce is None:
                if str(vid) != '[]':
                    beginning = str(vid).rfind('src=')+5
                    ending = str(vid).rfind(" width")-1
                    
                    url = str(vid)[beginning:ending]
                    temp2 = url.rfind("?")
                    temp3 = url.rfind(".",0,temp2)
                    
                    if temp2 == -1:
                    
                        extension = url[temp3:len(url)]
                
                    else:
                        extension = url[temp3:temp2]
                        
                    if platform.system()=="Windows":
                        name = 'img\\FanartSonic_' + str(i)+ extension
                    elif platform.system()=="Linux":
                        name = 'img/FanartSonic_' + str(i)+ extension
                    urllib.request.urlretrieve(url, main_path+name)
            
            else:
            
                img = sauce['src']
                temp2 = img.rfind("?")
                temp3 = img.rfind(".",0,temp2)
                if temp2 == -1:
                    
                    extension = img[temp3:len(img)]
                
                else:
                    extension = img[temp3:temp2]
                
                if platform.system()=="Windows":
                    name = 'img\\FanartSonic_' + str(i)+ extension
                elif platform.system()=="Linux":
                    name = 'img/FanartSonic_' + str(i)+ extension
                #name = 'img/FanartSonic_' + str(i)+ extension
                urllib.request.urlretrieve(img, main_path+name)
                         
    
'''      
#read the data
if platform.system()=="Windows":
    main_path = os.getcwd()+'\\'
elif platform.system()=="Linux":
    main_path = os.getcwd()+'/'

main_dir = main_path

index = main_dir.rfind('data')

main_dir = main_dir[0:index]

df = pd.read_csv(main_path+"dash_table.csv")

inf = 878
sup = 888

download_img(df, main_dir, inf, sup)'''