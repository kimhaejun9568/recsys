# README
**----------------------------Dataset ------------------**



├── yelp/(https://www.yelp.com/dataset)

│   ├── review.json   

│   ├── user.json   

│   ├── business.json  

│   ├── photos.json

│

├── embedding/( https://drive.google.com/file/d/1O6Izzvta2ehRN96rtPjaSyKH-3nq63Wo/view?usp=sharing )

│

├── data/ (https://drive.google.com/file/d/1SQPvuAr-oVzwDuokaJGm6AnCpJZVvuS3/view?usp=sharing)  

│

├── photos/((https://www.yelp.com/dataset))

│  

----------------------

Yelp Training(Main.py)

After aligning the dataset to the file location. 
You can run Main.py

---------------------------
Crawling(naver_kakao_data.ipynb)


This code serves to (1) crawl ratings data from Kakao Map and image/review data from Naver Map (2) match each other based on the name of the restaurant 

1. Download Chromedriver.exe to the matching version of your Chrome
2. Locate the Chromedriver.exe in the directory settings
3. Set directory settings
4. Run code

*Check Chrome Version
https://chromedriver.storage.googleapis.com/LATEST_RELEASE

*Download Chrome Driver
https://developer.chrome.com/docs/chromedriver/downloads?hl=ko

------------
Crawling preprocssing

----
Crawling prediction(Kakao_predict.py)

This code predicts the star rating based on Kakao Map reviews and compares it to the actual star rating based on the previously preprocessed Kakao dataset and photos.
