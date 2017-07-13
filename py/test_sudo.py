import subprocess

print('Please enter your sudo password, to complete the installation process:')
print(subprocess.check_output([
    'sudo', 'whoami']).decode('utf-8'))
