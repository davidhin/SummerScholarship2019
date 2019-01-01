#!/bin/bash

pid=`lsof -ti:8220`
kill $pid
