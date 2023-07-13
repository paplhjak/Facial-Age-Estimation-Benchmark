# Benchmarks & Databases

The benchmark is defined by a YAML file which contains a list of JSON databases. For each JSON database it defines N splits of the objects in the folders in the into trn/val/tst set. Assignment of the object into folders is defined in JSON database. 

The benchmark YAML to use is sepecified in 
```config['data']['benchmark']```

An example of benchmark YAML is as follows:
```
- database: path/face_database1.yaml  # path relative to the directory in which the benchmark.yaml is stored
  split:
     - trn: [0,1,2]
       val: [3]
       tst: [5]
     - trn: [1,2,3]
       val: [4]
       tst: [5]
- database path/face_database2.yaml
  split:
     - trn: [0,1,2]
       val: [3]
       tst: [5]
     - trn: [1,2,3]
       val: [4]
       tst: [5]
```