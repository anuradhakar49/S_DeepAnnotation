FROM cytomine/software-python3-base

COPY requirements.txt requirements.txt

RUN pip install \
  torch==1.10.0+cpu \
  torchvision==0.11.1+cpu \
  torchaudio==0.10.0+cpu \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html \
  && rm -rf /root/.cache/pip

RUN pip3 install opencv-python-headless==4.5.3.56

RUN pip install --no-cache-dir -r requirements.txt
#COPY src/ .

RUN mkdir images
ADD images /src/images
ADD . . 
RUN mkdir -p /src/outputs

ADD . /src/unet
ADD . . 
ADD . /src/utils
ADD . . 

#######unet components
ADD unet_model.py /src/unet_model.py
ADD unet_parts.py /src/unet_parts.py

#ADD inference_dataloader.py /src/inference_dataloader.py
#ADD help_functions.py /src/help_functions.py
#####################

ADD best_model_bg10.pth /src/best_model_bg10.pth
ADD app_model.py /src/app_model.py
ADD descriptor.json /src/descriptor.json
ADD . /src/preds


ENTRYPOINT ["python", "/src/app_model.py"]

