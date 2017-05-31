#!/bin/bash

g77 -B/usr/lib/i386-linux-gnu -o hphello1 hphello1.f -L. -lhp && ./hphello1

