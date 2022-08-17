NAS ���� mldisk�� �ּҰ� ����Ǿ�(mldisk2.sogang.ac.kr -> mldisk.sogang.ac.kr) ȣ��Ʈ�� �����̳ʿ��� ������ �ȵɰ�� ������ Ȯ���Ұ�

### �������
1. host���� `sudo apt install nfs-common`  

### ȣ��Ʈ���� ����Ʈ ����
1. `sudo vim /etc/fstab` ���� mldisk�� �ּҸ� ����
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
### docker-compose ���� �����̳ʿ��� ����Ʈ ����
1. ������Ʈ�� docker-compose.yml ���� volume�� �ּ� Ȯ��
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

