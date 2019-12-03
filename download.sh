conda activate expml

# https://stackoverflow.com/a/58141816
# pip install gdown

# Download cocolike
cd data
gdown https://drive.google.com/uc?id=1ueoqwcJ3imsN1xIm6z_1ZM4mwjX-DeI1
tar -xvf cocolike.tar.gz 

# Download samples
# gdown https://drive.google.com/uc?id=1dNYHuMmnOuU3H9fniuxGisYMOYeSUwbR

# If gdown doesn't work, try this
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QiqUCEt8uX0YdPUm7jCjEBu8M_LWz3Vk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt