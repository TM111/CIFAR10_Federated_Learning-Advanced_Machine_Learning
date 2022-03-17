# Federated Learning: where machine learning and data privacy can coexist

## Overview

Nowadays, a lot of data is generated but cannot be exploited due to its sensitive nature. For
instance, we can think of the data collected by the cameras or GPS sensors in our mobile phones,
or the performed ultrasounds and X-ray scans, or the data produced by the Internet of Things
(IoT). They are all of great value to the world of Big Data and Machine Learning (ML) applications,
but are also protected by privacy and, therefore, unusable by traditional methods.

Introduced in 2016 by Google, Federated Learning (FL) is a machine learning scenario born with
the aim of using privacy-protected data without violating the regulations in force. This framework
deals with learning a central server model in privacy-constrained scenarios, where data are
stored on multiple devices (i.e. the clients). Unlike the standard machine learning setting, here the
model has no direct access to the data, which never leave the clients’ devices: a fundamental
requirement for any application where users’ privacy must be preserved (e.g. medical records,
bank transactions).

## Goals

1. Replicate the experiments proposed by [this](https://arxiv.org/abs/2003.08082) on the CIFAR10 dataset.
2. Understand the contribution made by each parameter of the federated setting by proposing an experimental study.
3. Understand the importance of clients’ local data distribution in the federated scenario.
4. Understand the contribution of the normalization layers.
5. Make a contribution for solving existing issues.
