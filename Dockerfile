# FROM gcr.io/tensorflow/tensorflow:1.5.0-gpu
FROM docker.io/tensorflow/tensorflow:1.13.1-gpu-py3

RUN mv /usr/local/bin/pip /usr/local/bin/pip_2

RUN apt-get -y update && apt-get install -y python3-pip && pip3 install --upgrade pip

WORKDIR /maskr
ADD . /maskr

ENV PYTHONPATH="$PYTHONPATH:/maskr"

RUN pip3 install \
    pipenv \
    paperspace \
    tensorflow-gpu \
    Cython

RUN pip3 install \
    absl-py==0.7.1 \
    alabaster==0.7.12 \
    appnope==0.1.0 \
    astor==0.7.1 \
    attrs==19.1.0 \
    Babel==2.6.0 \
    backcall==0.1.0 \
    bleach==3.1.0 \
    certifi==2019.3.9 \
    chardet==3.0.4 \
    cycler==0.10.0 \
    decorator==4.4.0 \
    defusedxml==0.6.0 \
    docutils==0.14 \
    entrypoints==0.3 \
    gast==0.2.2 \
    grpcio==1.20.1 \
    h5py==2.9.0 \
    idna==2.8 \
    imageio==2.5.0 \
    imagesize==1.1.0 \
    imgaug==0.2.9 \
    ipykernel==5.1.0 \
    ipyparallel==6.2.3 \
    ipython==7.5.0 \
    ipython-genutils==0.2.0 \
    ipywidgets==7.4.2 \
    jedi==0.13.3 \
    Jinja2==2.10.1 \
    jsonschema==3.0.1 \
    jupyter-client==5.2.4 \
    jupyter-core==4.4.0 \
    Keras==2.2.4 \
    Keras-Applications==1.0.7 \
    Keras-Preprocessing==1.0.9 \
    kiwisolver==1.1.0 \
    Markdown==3.1 \
    MarkupSafe==1.1.1 \
    matplotlib==3.0.3 \
    mistune==0.8.4 \
    mock==3.0.5 \
    nbconvert==5.5.0 \
    nbformat==4.4.0 \
    networkx==2.3 \
    nose==1.3.7 \
    notebook==5.7.8 \
    numpy==1.16.3 \
    opencv-python==4.1.0.25 \
    packaging==19.0 \
    pandocfilters==1.4.2 \
    parso==0.4.0 \
    pexpect==4.7.0 \
    pickleshare==0.7.5 \
    Pillow==6.0.0 \
    prometheus-client==0.6.0 \
    prompt-toolkit==2.0.9 \
    protobuf==3.7.1 \
    ptyprocess==0.6.0 \
    pycocotools==2.0.0 \
    Pygments==2.4.0 \
    pyparsing==2.4.0 \
    pyrsistent==0.15.1 \
    python-dateutil==2.8.0 \
    pytz==2019.1 \
    PyWavelets==1.0.3 \
    PyYAML==5.1 \
    pyzmq==18.0.1 \
    qtconsole==4.4.4 \
    requests==2.21.0 \
    scikit-image==0.15.0 \
    scipy==1.2.1 \
    Send2Trash==1.5.0 \
    Shapely==1.6.4.post2 \
    six==1.12.0 \
    snowballstemmer==1.2.1 \
    Sphinx==2.0.1 \
    sphinxcontrib-applehelp==1.0.1 \
    sphinxcontrib-devhelp==1.0.1 \
    sphinxcontrib-htmlhelp==1.0.2 \
    sphinxcontrib-jsmath==1.0.1 \
    sphinxcontrib-qthelp==1.0.2 \
    sphinxcontrib-serializinghtml==1.1.3 \
    tensorboard==1.13.1 \
    tensorflow-gpu==1.13.1 \
    tensorflow-estimator==1.13.0 \
    termcolor==1.1.0 \
    terminado==0.8.2 \
    testpath==0.4.2 \
    tornado==6.0.2 \
    traitlets==4.3.2 \
    urllib3==1.24.3 \
    wcwidth==0.1.7 \
    webencodings==0.5.1 \
    Werkzeug==0.15.2 \
    widgetsnbextension==3.4.2

