FROM debian:12.11

# Configure apt sources
RUN echo "\nTypes: deb-src\n\
URIs: http://deb.debian.org/debian\n\
Suites: bookworm bookworm-updates\n\
Components: main\n\
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\n\
Types: deb-src\n\
URIs: http://deb.debian.org/debian-security\n\
Suites: bookworm-security\n\
Components: main\n\
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg" \
>> /etc/apt/sources.list.d/debian.sources
RUN apt update && apt install -y python3 python3-pip pipx vim git aptitude apt-rdepends&& \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure pipx uses the correct path
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /code

# Copy application files
COPY ./pyproject.toml /code/pyproject.toml
COPY ./uv.lock /code/uv.lock
COPY ./README.md /code/README.md
# Install dependencies
RUN pipx install uv

RUN apt update

# Set entrypoint
ENTRYPOINT ["bash", "-c"]
