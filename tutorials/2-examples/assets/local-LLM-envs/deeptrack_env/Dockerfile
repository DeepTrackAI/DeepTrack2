FROM continuumio/anaconda3
  
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y openssh-server sudo && \
    rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir deeptrack opencv-python && \
    conda install -y --quiet jupyterlab && \
    conda clean -afy


RUN useradd -m -s /bin/bash dev


RUN mkdir -p /home/dev/.ssh /run/sshd && \
    chmod 700 /home/dev/.ssh && \
    chown -R dev:dev /home/dev/.ssh


RUN sed -i \
    -e 's/^#PasswordAuthentication yes/PasswordAuthentication no/' \
    -e 's/^PasswordAuthentication yes/PasswordAuthentication no/' \
    -e 's/^#PubkeyAuthentication yes/PubkeyAuthentication yes/' \
    /etc/ssh/sshd_config

RUN mkdir -p /root/.ssh && \
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAfNHQ94iIubf0eG9S+E6NayfPuzXcQPWriuBMTPhWnW xhujia@GU.GU.SE@softmatterlab01-p.gu.gu.se" \
    > /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/authorized_keys && \
    chown -R root:root /root/.ssh

RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /home/dev/.bashrc

WORKDIR /a

EXPOSE 22 8888

CMD ["/bin/bash", "-c", "\
    /usr/sbin/sshd && \
    jupyter lab \
    --notebook-dir=/ \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
"]