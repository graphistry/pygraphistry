apt-get update

apt-get -y install curl git python clinfo nvidia-opencl-icd-352

export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
export ENV LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export C_INCLUDE_PATH=/usr/local/cuda/include

apt-get install -y mesa-utils libxrandr-dev libxcursor-dev libglew-dev libglu1-mesa-dev libfreeimage-dev libxinerama-dev xinput libxi-dev cmake unzip

for key in 9554F04D7259F04124DE6B476D5A82AC7E37093B 94AE36675C464D64BAFA68DD7434390BDBE9B9C5 0034A06D9D9B0064CE8ADF6BF1747F4AD2306D93 FD3A5288F042B6850C66B31F09FE44734EB7990E 71DCFD284A79C3B38668286BC97EC7A07EDE3FC1 DD8F2338BAE7501E3DD5AC78C273792F7D83545D B9AE9905FFD7803F25714661B63B535A4C206CA9 C4F0DFFF4E8C1A8236409D08E73BC641CC11F4C8 ; do gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$key"; done

export NODE_VERSION=0.12.12
export NPM_VERSION=2.14.1

curl -SLO "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz"
curl -SLO "https://nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt.asc"
gpg --batch --decrypt --output SHASUMS256.txt SHASUMS256.txt.asc
grep " node-v$NODE_VERSION-linux-x64.tar.xz\$" SHASUMS256.txt | sha256sum -c -
tar -xJf "node-v$NODE_VERSION-linux-x64.tar.xz" -C /usr/local --strip-components=1
rm "node-v$NODE_VERSION-linux-x64.tar.xz" SHASUMS256.txt.asc SHASUMS256.txt
npm install -g npm@"$NPM_VERSION"
npm cache clear

cd /
git clone https://github.com/glfw/glfw.git
cd /glfw
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=TRUE && make && make install

cd /
git clone https://github.com/graphistry/cljs.git
cd /cljs
npm link
npm run test
cd examples/convolutionDemo
npm link
npm start
