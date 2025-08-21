#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config.common.leaves.leaf_actions as la


def get_health_manager_node_actions():
  health_check_node =  la.LeafAction(repo='health_manager',
              node_name='device_health_check_node',
              action_name='health_check_action',
              input_refs=[],
              description='health_check description'
  )

  return [
    health_check_node,
  ]

