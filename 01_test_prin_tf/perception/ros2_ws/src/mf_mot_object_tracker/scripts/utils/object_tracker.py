import numpy as np

class ObjectTracker:
  """Utility class for tracking objects across frames"""
  
  def __init__(self, tracking_distance_threshold=0.1, max_missing_frames=10):
    self.tracking_distance_threshold = tracking_distance_threshold
    self.max_missing_frames = max_missing_frames
    
    # Tracking system
    self.fruit_tracks = {}  # track_id -> track info
    self.next_track_id = 0
    self.frame_count = 0
    
  def associate_detections_with_tracks(self, detections):
    """
    Associate new detections with existing tracks based on position similarity
    Only tracks detections that have valid 3D position information
    Args:
      detections: List of detection dictionaries with 'position_3d' key
    Returns:
      List of (track_id, detection) tuples
    """
    self.frame_count += 1
    
    # Update existing tracks (mark as not updated)
    for track_id in self.fruit_tracks:
      self.fruit_tracks[track_id]['updated'] = False
      self.fruit_tracks[track_id]['missing_frames'] += 1
    
    associated_pairs = []
    
    # Filter detections - only process those with valid 3D position
    trackable_detections = []
    non_trackable_detections = []
    
    for det_idx, detection in enumerate(detections):
      if detection['position_3d'] is not None:
        trackable_detections.append((det_idx, detection))
      else:
        non_trackable_detections.append((det_idx, detection))
    
    print(f"  Tracking: {len(trackable_detections)} detections with depth, {len(non_trackable_detections)} without depth")
    
    unassociated_detections = list(range(len(trackable_detections)))
    
    # Try to associate each trackable detection with existing tracks
    for i, (det_idx, detection) in enumerate(trackable_detections):
      if i not in unassociated_detections:
        continue
        
      det_pos = np.array(detection['position_3d'])
      best_track_id = None
      best_distance = float('inf')
      
      # Find closest track
      for track_id, track_info in self.fruit_tracks.items():
        if track_info['updated']:  # Already associated
          continue
          
        track_pos = np.array(track_info['last_position'])
        distance = np.linalg.norm(det_pos - track_pos)
        
        if distance < self.tracking_distance_threshold and distance < best_distance:
          best_distance = distance
          best_track_id = track_id
      
      if best_track_id is not None:
        # Associate detection with existing track
        self.fruit_tracks[best_track_id]['last_position'] = det_pos
        self.fruit_tracks[best_track_id]['updated'] = True
        self.fruit_tracks[best_track_id]['missing_frames'] = 0
        associated_pairs.append((best_track_id, detection))
        unassociated_detections.remove(i)
        print(f"  Associated detection {det_idx} with existing track {best_track_id} (dist: {best_distance:.3f}m)")
    
    # Create new tracks for unassociated trackable detections
    for i in unassociated_detections:
      det_idx, detection = trackable_detections[i]
      new_track_id = self.next_track_id
      self.next_track_id += 1
      
      self.fruit_tracks[new_track_id] = {
        'last_position': np.array(detection['position_3d']),
        'updated': True,
        'missing_frames': 0,
        'created_frame': self.frame_count
      }
      
      associated_pairs.append((new_track_id, detection))
      print(f"  Created new track {new_track_id} for detection {det_idx}")
    
    # Handle non-trackable detections (without depth) - assign temporary IDs for visualization only
    for det_idx, detection in non_trackable_detections:
      # Use negative IDs for non-tracked detections to distinguish them
      temp_id = -(det_idx + 1)  # -1, -2, -3, etc.
      associated_pairs.append((temp_id, detection))
      print(f"  Detection {det_idx}: No depth - assigned temporary ID {temp_id} (visualization only)")
    
    # Remove tracks that have been missing for too long
    tracks_to_remove = []
    for track_id, track_info in self.fruit_tracks.items():
      if track_info['missing_frames'] > self.max_missing_frames:
        tracks_to_remove.append(track_id)
        print(f"  Removing track {track_id} (missing for {track_info['missing_frames']} frames)")
    
    for track_id in tracks_to_remove:
      del self.fruit_tracks[track_id]
    
    print(f"  Frame {self.frame_count}: {len([p for p in associated_pairs if p[0] > 0])} tracked, {len([p for p in associated_pairs if p[0] < 0])} untracked, {len(self.fruit_tracks)} active tracks")
    
    return associated_pairs

  def get_active_tracks_count(self):
    """Get number of active tracks"""
    return len(self.fruit_tracks) 