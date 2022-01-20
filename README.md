# Federated Learning: where machine learning and data privacy can coexist
##Overview
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
In this project, you will become familiar with the federated scenario and its standard architecture.
Once implemented your baseline, you will analyze the variations that occur by modifying the
value of the parameters specific to this framework, assessing the most effective ones. Finally,
basing on the identified problems of the resulting model, you will propose a possible solution to
address one of them.
##Goals
1. To get familiar with Federated Learning and its main algorithms and architecture.
2. To replicate the experiments proposed by [3] on the CIFAR10 dataset.
3. To understand the contribution made by each parameter of the federated setting by proposing an experimental study.
4. To understand the importance of clients’ local data distribution in the federated scenario.
5. To understand the contribution of the normalization layers.
6. To make your contribution for solving existing issues.

