#!/bin/bash

for model in `ls *.nd`
do
    echo $model
    taup_create $model
done
