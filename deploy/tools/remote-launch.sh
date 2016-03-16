echo '
SSH_ENV="$HOME/.ssh/environment"

function start_agent {
    echo "Initialising new SSH agent..."
    /usr/bin/ssh-agent | sed 's/^echo/#echo/' > "${SSH_ENV}"
    echo succeeded
    chmod 600 "${SSH_ENV}"
    . "${SSH_ENV}" > /dev/null
    /usr/bin/ssh-add;
}

# Source SSH settings, if applicable

if [ -f "${SSH_ENV}" ]; then
    . "${SSH_ENV}" > /dev/null
    #ps ${SSH_AGENT_PID} doesn't work under cywgin
    ps -ef | grep ${SSH_AGENT_PID} | grep ssh-agent$ > /dev/null || {
        start_agent;
    }
else
    start_agent;
fi
' > ~/.bash_profile
# via http://mah.everybody.org/docs/ssh

source ~/.bash_profile

sudo apt-get install git

mkdir wholly-innocuous

echo H4sIAJg0a1YCA22XtxLjSg5Fc37Fy1lT9C7YgF70RvQZvRW96L5+NW/T7airOgEa5wIXf/78DifKivmP+2b/sV0lYD3xH02M/z78AQxFUTVW4VhW49laZM/voE018eEXeIx37BuDASmM9TLxl/yhFlA1yJf04qMOi3DtFgJAErpdlsxmte429Ci5A2Pb+7TT8SoUzM55iEkJ0RAbB7uQHlSmhdBfkb8VxMRuZk0HgH+ExOYghK4QSIWWpaRbWDZm9tWxmpOaiOWW0ihbTunu/YusqjLqR6xk0rf09rJYwwCh82Xo8floaF5PYGeNJcRZpwkJ+0lpEMQP4naWQzww2yYJNjjFh7cdPNRzoeYGbeWB14y9kNNSh2FBTBeyldDR8BKPOQjOGLJg0DfXjj5k61pFNU40+ThL3+BrO8p9D9ZzAC6WcQboGy0pGynLg1ZebQ0U1m+L5R8Eu4AVlUgwf3EoX7Z1dXKn90o+xfFZec0d4wKQBTZfQ7JasddIcucxYaXH1xV3rqETp+iSJ16opdVoIrr4jF06mA0ZxIscdb1kXH4J7PeeIjYfZuLH4KDAU0HTpnC+qphMIJmFJctZwVQlfXtHLxof8z5U+mEU3ZHeWM9CExBO0KTKIunR8eyoSdd6ezo0Nq/tkH6xz6qGPH5VOsMa0ZZzykdiwzDKJqWPpV+CgwxomKhjx1AXT2RCb1b8RYckehoUYJTR6ESx/c6zLFae10CAAacU0WGwhFCqoNYneQsC7+6+EIFqZMMeqO93NG6erjGJ/kG2tHS2fNn8PCrhY/F55OemERTznNkPeSI0z54iC7DOX4TZysyqdb5I776UuVAPpjk2Mi5LLh1CQ2i902FGLmpwedXpj2WEw+ODxBfjeKDqP9Oha++Hj7J3vy79mi/yst8uJH/fFylnpbTji543PcFL7FZNxHzuMRr2LsHecSgAy+GorQIRjRbAox7SqdzlXE4QKdPpKojDQXfRwrvYuzprTb1otaRT737hupL90EEtAQcq2nRuWU84pe3NptiSVaaK8pMruXr1SfmM/pVtikOlxrXUqe2qfCPqs13NCl2WwQCvBr7A8bjtncfeWgrnhm+vprMyg+zLx/B01zzbL2oXoHAzwfU795N0oJS/cnkxHugDQMmFnkdRaKlGdxEaBV6Meh/p8T8DHCQpndYc1z9LVMBZfyMZ2RgzrNVlReeIVm1zCGhklMloKKl3CA5iaVBfN0ANJ+uXzK00sIDox6BO/4bEKkVyVNTa2+1MA/mEZ4w1rA144Oub3KQ5BVaD0vinkW/DWyaKSlYuduLwqVieq2t0wB8zsPngeOxWSWORt4RC9u8ZkLla4Rn1DafIReKZo0KfAoslmMKM2817MrsHz7GM21ZF+4YpyC+rNsL2oIIkLKfDNwA/RvV2WnI3dozxo1DGv09KK2N5JrAVSLo8f99U1zhq3FsNObGZhGZTMVd8gelHJvRAxmOjUHEyFtYdqKxThRJ7jflPJ+DhhGjpN7FzBxPerdbhCVV/6b/kOiILdflFpz0P2Nyb77+yUgQohar+wcVIqmUJETRvjI1W2zsHk2W3+r6/L/6dVVR4n8IeKr+2tMW6iwKMQQulY4ARbUabEJbx3YCjZA3EdzNhNUfk7AL/3tDGIWpDQXpMi4RKSNQVk6I7o4BoL73ykfadOd6zZqSpFz6rx8aOUo4Hn49VtDan7GAjg1p76sq8XxHprms4ix5iHoNAD3uNn5+yx/kCckSTcL2FfZ8xVau5LIJ/TaMucFWDXJtDa7+MsvqtVYd/2r611B+8BWLWOWqa4xI87QPRiWnas3NT6tOByFAYao73sE6UanuhmSDolmslE7PkK/DeknCsxQ9l/nQqiUjtJE3z838/fImlwltX+S7hQmRdOxaX4oG20mnXHnyHwzd9sld5vjpxxQCtrvpt/zTtu1xx60VEU1hLEf8lj0Zyi4hdMnssIkh37i9HOXlUPzOcZhk7XDboha4GOP4tMpPWdBfvmXjuRN76g4ZoCLTIE5KI52bgdOvcKf10PyLfecFIGUkkJOL8Wb2WA8jYJAdtODrPpisSgaf5mj84Bf/EgdvS68Wh1KcvipbAGqehyaPWgsvl/OHXKE4IWVBg07n8u1iZwFblD2emh53rKILuO10IfQkn6w81ig5lesKogN4PT5iTS91Jd7LaXBErIM2bTISkbjeipxnH60lLeb2rj6nWFcMT69oo73Wu8uRfhDmu5D7y5uCse0LXyIA3cHe5kbY0RptBMehUSX5WW9bBH/3sUOr1E67XR7HXsnoFPKQzll/qImX75n0tOprXLNCkXDJdwR6y8xNx4OvYAxV08gCjLAVyHL5mqO7NC/Wrf8RVauqIMRZEEOkhvV47DUcAvmQoiKvZ2WQqFVJXheGIpdUwuaSDSE7YRH48pZaqYqpepYPzeW6RdSsfGVXZzTmygDJINWKV8DJFSUhgUKHDSYzdnKGllxqMMejf67aqP9FyBVbufkmTGKJIKuJM1dBivzKmD10t/if/DTXB3Io6+2phqlDOHXqVpa5Vzh8F3QjCi3Icrcgteu1Hm+gwD1F/UwgD7Lhi2Onn0mTYurM+616cBvnCwfFW2VewsKpJZp3zps/UbY3Bsf9EOgfyy/aQ9D7cCXCVKXFtyxKIJZWoNCUgLWvZgsd0tZwDOxI3u+y/9uybsqgSqiYT13Re0mWYJeT/RAIEk1Kg5OE2oIwarl178idKaPzOGlWC5bRyNAblt2TG1gSP2p85ka03yZ+BRwuhzRkpwMSbMFBCmhP7HGmn3Ns4D3ZS8kLLIkY7JN6He4X8jaFqsiLdQN8yLx7XUoGreKFLE6hADb+6wnO32u2++FsLaBKHoIfFzw+f4Uh9TJ0Q85RoQJRuMsGteuXRhpSfWoX98SZAZrZZOINxL1ln9ScIkaS0T3Cn1x4TFXq+rkWOtc1MU1FZgvsh/knLNFVR0vqxNXsBqL8ZfYSCbnUObiT6z/xRaQk2nvGhERdXsnmmtm93qjTuJu5JmStn0FtGV6VJZcI8wkDhcMYMDR4yChSd2C/tzm4bccD4uirRE9OUbOBh21GnZsyHw6Da3Fo/cRK1bMHi+qXAKPfQ2UUtpl5vktZaNiN7eD6GFum2L3x+YG1brNXwYIkw118oa68sP9F2TPvbbS4bqPSK/zlAeDEKfO+Y+Lu/el8ua/DzTHwtEIPwMPyGoMuxenenGStLtDrqgscUednwkoBJRZnmnp7ApTz2PYj20j+vRZ3SNE5yi6GSa/K+bJiolbpwP9NNW9M1LuPIYNN/gH9XDtEU/v8q8l/kAASmqwwAAA== | base64 --decode | gzip -cd > wholly-innocuous/nothing-to-see-here

ssh-add wholly-innocuous/nothing-to-see-here

git clone git@github.com:graphistry/deploy.git

cd deploy

chmod 600 tools/ansible_id_rsa.pem

sudo apt-get install python-pip

sudo pip install ansible==1.9.4

sudo pip install jinja2

ssh -o StrictHostKeyChecking=no -i tools/ansible_id_rsa.pem ubuntu@<<<my-host>>> whoami | grep ubuntu

ansible-playbook site.yml -i <<<my-host-file>>>

