import json
from abc import ABC
from abc import abstractmethod
from itertools import chain
from typing import List

import dns.resolver

from exaflow.controller import DeploymentType


class WorkersAddresses(ABC):
    @abstractmethod
    def __init__(self):
        self._socket_addresses = None

    @property
    def socket_addresses(self) -> List[str]:
        return self._socket_addresses


class FileWorkersAddresses(WorkersAddresses):
    def __init__(self, workers_configs):
        with open(workers_configs.config_file) as fp:
            self._socket_addresses = json.load(fp)


class DNSWorkersAddresses(WorkersAddresses):
    def __init__(self, workers_configs):
        dns_names = [n.strip() for n in workers_configs.dns.split(",") if n.strip()]
        ip_lists = (dns.resolver.resolve(name, "A", search=True) for name in dns_names)

        # flatten the iterator‑of‑iterators and de‑duplicate
        uniq_ips = {rr.address for rr in chain.from_iterable(ip_lists)}
        self._socket_addresses = [f"{ip}:{workers_configs.port}" for ip in uniq_ips]


class WorkersAddressesFactory:
    def __init__(self, depl_type: DeploymentType, workers_configs):
        self.depl_type = depl_type
        self.workers_configs = workers_configs

    def get_workers_addresses(self) -> WorkersAddresses:
        if self.depl_type == DeploymentType.LOCAL:
            return FileWorkersAddresses(self.workers_configs)

        if self.depl_type == DeploymentType.KUBERNETES:
            return DNSWorkersAddresses(self.workers_configs)

        raise ValueError(
            f"DeploymentType can be one of the following: {[t.value for t in DeploymentType]}, "
            f"value provided: '{self.depl_type}'"
        )
