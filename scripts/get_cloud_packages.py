#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from time import sleep

# —— 上面汇总的所有包名（按需增删） ——
packages = [
    "openssh-server",
    "cloud-init",
    "ntp",
    "chrony",
    "docker.io",
    "containerd",
    "runc",
    "containernetworking-plugins",
    "kubelet",
    "kubectl",
    "kubeadm",
    "cri-o",
    "slurm-wlm",
    "cloudendure-agent",
    "mysql-server",
    "mariadb-server",
    "postgresql",
    "redis-server",
    "memcached",
    "mongodb-clients",
    "mongodb-org",
    "cassandra",
    "kafka",
    "hadoop",
    "spark",
    "hive",
    "presto-server",
    "trino-server",
    "flink",
    "apache-airflow",
    "elasticsearch",
    "opensearch",
    "activemq",
    "rabbitmq-server",
    "camel",
    "nifi",
    "python3-ebcli",
    "lambci-runtime",
    "fabric-peer",
    "fabric-ca-client",
    "tensorflow",
    "pytorch",
    "mxnet",
    "nvidia-container-toolkit",
]

BASE_URL    = "https://tracker.debian.org/pkg/{}"
OUT_FILE    = "non404.txt"
TIMEOUT_SEC = 5
PAUSE_SEC   = 0.1   # 每次请求后稍作停顿，避免被目标服务器限流

def main():
    non404 = []

    print(f"Checking {len(packages)} packages, timeout={TIMEOUT_SEC}s ...\n")
    for pkg in packages:
        url = BASE_URL.format(pkg)
        try:
            resp = requests.head(url, allow_redirects=True, timeout=TIMEOUT_SEC)
            code = resp.status_code
        except Exception as e:
            print(f"[ERR] {pkg}: {e}")
            sleep(PAUSE_SEC)
            continue

        if code == 404:
            print(f"[404] {pkg}")
        else:
            print(f"[{code}] {pkg}")
            non404.append(pkg)

        sleep(PAUSE_SEC)

    # 写文件
    with open(OUT_FILE, "w", encoding="utf-8") as fw:
        for pkg in non404:
            fw.write(pkg + "\n")

    print(f"\nDone. {len(non404)} / {len(packages)} packages are non-404.")
    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()