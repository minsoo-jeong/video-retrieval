NAS 서버 mldisk의 주소가 변경되어(mldisk2.sogang.ac.kr -> mldisk.sogang.ac.kr) 호스트나 컨테이너에서 연결이 안될경우 다음을 확인할것

### 공통사항
1. host에서 `sudo apt install nfs-common`  

### 호스트에서 마운트 에러
1. `sudo vim /etc/fstab` 에서 mldisk의 주소를 변경
``` bash
# /etc/fstab
... 
/swapfile                                 none            swap    sw              0       0
...
mldisk.sogang.ac.kr:/volume2/nfs_shared /media/mmlab/mldisk/nfs_shared auto nosuid,nodev,nofail,x-gvfs-show 0 0
mldisk.sogang.ac.kr:/volume3/nfs_shared_ /media/mmlab/mldisk/nfs_shared_ auto nosuid,nodev,nofail,x-gvfs-show 0 0
mlsun.sogang.ac.kr:/volume1/nfs_shared /media/mmlab/mlsun/nfs_shared auto nosuid,nodev,nofail,x-gvfs-show 0 0 
 ```
2. `sudo mount -a`
### docker-compose 사용시 컨테이너에서 마운트 에러
1. 프로젝트의 docker-compose.yml 에서 volume의 주소 확인
```
# docker-compose.yml
services:
...
volumes:
  - type: volume 
    source: nfs_shared
    target: /nfs_shared
    volume: 
      nocopy: true
...
volumes:
  nfs_shared:
    driver_opts:
      type: "nfs"
      o: "addr=mldisk.sogang.ac.kr,nolock,soft,rw"
      device: ":/volume2/nfs_shared"
  ```

