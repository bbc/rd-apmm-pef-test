# Copyright (c) 2021 BBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SOURCES=$(wildcard src/*.c)
HEADERS=$(wildcard src/*.h)
OBJS=$(patsubst %.c,%.o,${SOURCES})

CFLAGS=-g -msse -msse2 -msse3 -mavx -mavx2
LDFLAGS=

all: pef-test

%.o: %.c
	gcc -c ${CFLAGS} -o $@ $<

pef-test: ${OBJS}
	gcc ${CFLAGS} ${LDFLAGS} -o $@ ${OBJS}

clean:
	-rm -rf pef-test
	-rm -rf src/*.o
