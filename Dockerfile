FROM python:3

ADD yelp_clustering.py /

RUN pip install pandas==1.0.0 \
	matplotlib==2.1.2 \
	seaborn==0.10.0 \
	scikit-learn==0.22.2

CMD [ "python", "./yelp_clustering.py" ]