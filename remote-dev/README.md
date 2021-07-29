## Configure SSH access to your remote server (optional one-time step)

### 1. Create an SSH Shortcut

If you are constantly needing to SSH into multiple servers, it can real daunting to remember all the different usernames, hostnames, IP addresses, and even sometimes custom private keys to connect to them. It's actually extremely easy to create command line shortcuts to solve this problem. More info here: https://linuxize.com/post/using-the-ssh-config-file/

Edit SSH config:
```sh
nano ~/.ssh/config
```

Add Server Info to the end of file:

```
Host gpu1
     HostName XX.XXX.XXX.XX
     User root
```

In this example shortcut is `gpu1`. `HostName` - is an ip-address to your server.

### 2. Set up public key authentication

ssh-copy-id installs an SSH key on a server as an authorized key. Its purpose is to provision access without requiring a password for each login. This facilitates automated, passwordless logins and single sign-on using the SSH protocol. More info here: https://www.ssh.com/ssh/copy-id


Check that you have SSH keys:
```sh
cat ~/.ssh/id_rsa.pub
```

If you see message `No such file or directory` then run following command to generate as SSH key:
```sh
ssh-keygen
```

Authorize you SSH key on remote server. I will need to enter remote server password
```sh
ssh-copy-id -i ~/.ssh/id_rsa.pub gpu1
```

### 3. Check SSH access

Now you can type `ssh gpu1` in terminal to connect to your remote server quickly.

<img src="https://i.imgur.com/8OZH2Xw.png"/>


### 4. Clone repo and checkout to dev branch

```sh
git clone https://github.com/supervisely-ecosystem/unet
cd unet

# checkout to dev branch (optional)
git branch -a
# you will see something like that:
#* master
#  remotes/origin/HEAD -> origin/master
#  remotes/origin/init-docker-and-dev
#  remotes/origin/master
git checkout --track origin/init-docker-and-dev
# now you at the target branch
git pull
git status
```

### 4. Build dockerimage for your future app
```sh
cd docker && ./step-01-build-local-image.sh
```

### 5. Start remote development (optional for development on remote server)
```sh
cd remote-dev
```
