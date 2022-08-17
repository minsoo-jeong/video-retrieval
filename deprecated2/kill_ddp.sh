#!/usr/bin/env bash
#set -x
pid=$(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}' | tr '\n' ' ')
if [ "$(wc -w <<< $pid)" -gt 0 ]
then
  echo "kill process $pid"
  kill -9 $pid
fi

