from __future__ import print_function, division
from os import path
from os.path import join
import traceback
import yaml
import argparse
from gi.repository import Gtk, GLib
from travispy import TravisPy

try:
    from gi.repository import AppIndicator3 as AppIndicator
except:
    from gi.repository import AppIndicator


script_dir = path.dirname(path.realpath(__file__))


class Systray(object):
    def __init__(self):
        self.ind = AppIndicator.Indicator.new(
                            "indicator-cpuspeed",
                            "onboard-mono",
                            AppIndicator.IndicatorCategory.HARDWARE)
        self.ind.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.menu = Gtk.Menu()

        # this is for exiting the app
        item = Gtk.MenuItem()
        item.set_label("Exit")
        item.connect("activate", self.handler_menu_exit)
        item.show()
        self.menu.append(item)

        self.menu.show()
        self.ind.set_menu(self.menu)

        # initialize cpu speed display
        self.instance_items = []
        self.update()
        # then start updating every 2 seconds
        # http://developer.gnome.org/pygobject/stable/glib-functions.html#function-glib--timeout-add-seconds
        GLib.timeout_add_seconds(10, self.handler_timeout)

    def handler_menu_exit(self, evt):
        Gtk.main_quit()

    def handler_timeout(self):
        """This will be called every few seconds by the GLib.timeout.
        """
        self.update()
        # return True so that we get called again
        # returning False will make the timeout stop
        return True

    def update(self):
        label = ''
        try:
            # message = ''
            for slug in config['repos_to_watch']:
                repo = t.repo(slug)
                build_state = repo.last_build_state
                if build_state not in ['passed', 'failed']:
                    if label != '':
                        label += ' '
                    label += slug.split('/')[1] + ':' + build_state
            # print(repo.last_build_state)
            pass
        except Exception as e:
            label = 'exception occurred'
            try:
                print(traceback.format_exc())
            except:
                print('exception in exception :-P')
        self.ind.set_label(label, "")

    def main(self):
        Gtk.main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', default=join(script_dir, 'travis.yaml'))
    args = parser.parse_args()
    with open(args.configfile, 'r') as f:
        config = yaml.load(f)
    api_token = config['api_token']
    t = TravisPy.github_auth(api_token)
    user = t.user()
    print(user.login)
    # for repo in t.repos(member=user.login):
    #     print(repo.slug, repo.last_build_state)
    ind = Systray()
    ind.main()
