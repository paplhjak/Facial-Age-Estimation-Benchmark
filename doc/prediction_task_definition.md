# Prediction task definition

The library supports multi-head architecture. Each head defines a single prediction problem. E.g. the following example considers prediction of age and gender:
```
heads:
   - tag: "age"
     attribute: "age"
     label: [[0],[1],...,[90,91,92,...,100]]
     classes: [0,1,...,100]
     weight: 0.5
     metric: ['mae','cs5']
     visual_metric: ['mae]
   - tag: "gender"
     attribute: "gender"
     label: ["M","F"]
     classes: ["male","female"]
     weight: 0.5
     metric: ['0/1']
     visual_metric: ['cm']
```
- "tag" is a name of the prediction task. 
- "attribute" is an item in JSON the content of which is mapped to the cannonical label (labels used by the NN head).
- "label" maps the JSON's attribute value to a cannonical label which is from [0,1,...,len(label)-1]. 
- "classes" (optional) maps the cannonical label to the output label, i.e. human interpretable output. It is used only in the "predict.py".
- "metric" defines performance measures that are computed by "evaluate.py". metric[0] is the metric minimized in the validation stage. In case of multiple heads, the validation stage uses a weighted combination of the metrics.
- "weight" is the head's  contribution to the training loss and the validation metric.
- "visual_metric" are computed and visualuzed by "evaluate.py". Paramerers of the visualization metric cn be entered as follows:
```
   - tag: "adult"
     attribute: "age"
     labels: !include labels_adult.yaml
     classes: ["young","adult"]
     weight: 0.33
     metric: ['0/1']
     visual_metric:
        - tag: 'prc'
          target_class: [0]
          target_class_prior: [0.5,0.8,0.9]
        - tag: 'roc'
        - tag: 'cm'

```
