#!/bin/sh
ps -aux|grep tensorflow|awk '{print $2}'|xargs kill -9