FROM jiaxy/centos-python3.10:v2

RUN pip install poetry

RUN yum install mesa-libGL-devel mesa-libGLU-devel freeglut-devel -y

RUN mkdir -p /workspace_temp/Langchain-Chatchat
COPY ./Langchain-Chatchat /workspace_temp/Langchain-Chatchat
COPY entrypoint.sh /workspace_temp/
ENV CHATCHAT_ROOT=/parth/to/chatchat_data
WORKDIR /workspace_temp/Langchain-Chatchat/libs/chatchat-server
#RUN poetry install --with lint,test -E xinference
RUN poetry update
RUN poetry build

COPY ./nltk_data-gh-pages/packages/chunkers/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/corpora/* /parth/to/chatchat_data/data/nltk_data/corpora/*
COPY ./nltk_data-gh-pages/packages/grammars/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/help/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/misc/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/models/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/sentiment/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/stemmers/ /parth/to/chatchat_data/data/nltk_data/
COPY ./nltk_data-gh-pages/packages/taggers/* /parth/to/chatchat_data/data/nltk_data/taggers/
COPY ./nltk_data-gh-pages/packages/tokenizers/* /parth/to/chatchat_data/data/nltk_data/tokenizers/

RUN mkdir /workspace
RUN mkdir /parth1
#ENTRYPOINT ["/workspace_temp/entrypoint.sh"]
#RUN cp -r /workspace_temp/Langchain-Chatchat /workspace/Langchain-Chatchat
#RUN cp -r /parth /parth1

