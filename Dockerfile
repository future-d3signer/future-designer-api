FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and miniconda
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git ca-certificates && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN conda create -n myenv python=3.11 -y && \
    echo "Conda environment 'myenv' created."

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN python --version && pip --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git && \
    cd sam2 && \
    pip install . && \
    cd .. && \
    rm -rf sam2

COPY ./app ./app 

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]