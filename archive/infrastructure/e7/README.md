# E7 Distributed Rehearsal

The local cluster proves packaging, logical scheduling, deterministic task
histories, and Ray worker-process recovery. It does not prove physical
multi-host behavior and cannot complete E7.

## Local Docker rehearsal

Use the Python 3.12 Ray environment for the driver:

```powershell
$env:GEODE_CACHE_DIR = 'D:\geode-ml\data\cache'
docker compose -f infrastructure/e7/docker-compose.yml up -d --build --scale ray-worker=2
& 'D:\geode-ml\envs\cg-moe-ray\Scripts\python.exe' `
  -m experiments.e2e.run_e7_local_cluster_rehearsal
& 'D:\geode-ml\envs\cg-moe-ray\Scripts\python.exe' `
   -m experiments.e2e.run_e7_domainnet_local_small
docker compose -f infrastructure/e7/docker-compose.yml down
```

The resulting `e7_local_cluster_rehearsal.json` must report
`local_simulation_gate_passed: true` and `e7_gate_passed: false`.
The `e7_domainnet_local_small.json` artifact additionally requires exact
feature replay over a bounded real-DomainNet, all-domain episode. It is a
systems qualification using a nearest-centroid control, not a flagship model
or performance claim.

## Cloud qualification

1. Build and publish `infrastructure/e7/Dockerfile` to the target registry.
2. Install the KubeRay operator and provision an RWX `geode-cache-rwx` claim,
   or replace the volume with the target cloud's object-storage adapter.
3. Replace `geode-e7:latest` with the immutable image digest.
4. Apply `kuberay-cluster.yaml`; its topology constraint requires worker pods
   on distinct Kubernetes hosts.
5. Record pod UID-to-`spec.nodeName` placement before and after deleting one
   worker node, then run the full DomainNet cluster-small qualification.

Terraform should provision the provider-specific network, Kubernetes cluster,
node pools, registry, IAM, and storage. Those resources intentionally remain
outside this provider-neutral template until a cloud provider is selected.
