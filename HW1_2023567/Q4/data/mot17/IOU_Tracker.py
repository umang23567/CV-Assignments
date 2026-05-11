# import numpy as np

# class IOUTracker:
#     def __init__(self, iou_threshold=0.7):
#         # 1. Initialize the IOU threshold and required variables like trackers and next_id.
#         self.next_id=0
#         pass

#     def _compute_iou(self, box1, box2):
#         """
#         2. Implement the IOU computation between two bounding boxes.
#         - This involves calculating the intersection and union areas of the boxes.
#         - Return the IOU score (intersection / union).
#         """
#         pass

#     def update(self, detections):
#         """
#         3. Implement the update method to update the trackers with new detections.
#         - Perform greedy matching using IOU: compare each tracker with new detections.
#         - If IOU > threshold, match the detection to the tracker.
#         - If no match is found for a detection, create a new tracker.
#         - Remove trackers that don't match any detection from the previous frame.
#         - Input Format : [xmax,ymin,xmax,ymax,score,class]
#         - Return the updated list of trackers in the format: [xmin, ymin, xmax, ymax, track id, class, score].
#         """
#         pass

import numpy as np

class IOUTracker:
    def __init__(self, iou_threshold=0.3):
        # 1. Initialize the IOU threshold and required variables like trackers and next_id.
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = []  # Stores: {'bbox': [x1,y1,x2,y2], 'id': int, 'cls': int, 'score': float}

    def _compute_iou(self, box1, box2):
        """
        2. Implement the IOU computation between two bounding boxes.
        - This involves calculating the intersection and union areas of the boxes.
        - Return the IOU score (intersection / union).
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, (x2_inter - x1_inter) * (y2_inter - y1_inter))

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area > 0 :
            return inter_area / union_area 
        else:
            return 0

    def update(self, detections):
        """
        3. Implement the update method to update the trackers with new detections.
        - Perform greedy matching using IOU: compare each tracker with new detections.
        - If IOU > threshold, match the detection to the tracker.
        - If no match is found for a detection, create a new tracker.
        - Remove trackers that don't match any detection from the previous frame.
        - Input Format : [xmax,ymin,xmax,ymax,score,class]
        - Return the updated list of trackers in the format: [xmin, ymin, xmax, ymax, track id, class, score].
        """
        new_tracks = []
        matched_indices = set()

        # Greedy matching- Compare existing tracks to new detections
        for track in self.tracks:
            best_iou = -1
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_indices:
                    continue
                
                iou = self._compute_iou(track['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            # When match found above threshold update the track
            if best_iou >= self.iou_threshold:
                matched_indices.add(best_det_idx)
                det = detections[best_det_idx]
                new_tracks.append({
                    'bbox': det[:4],
                    'id': track['id'],
                    'cls': int(det[5]),
                    'score': det[4]
                })

        # Create new trackers for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_indices:
                new_tracks.append({
                    'bbox': det[:4],
                    'id': self.next_id,
                    'cls': int(det[5]),
                    'score': det[4]
                })
                self.next_id += 1

        self.tracks = new_tracks
        
        # Format for output
        results = []
        for t in self.tracks:
            results.append([*t['bbox'], t['id'], t['cls'], t['score']])
            
        return np.array(results) if results else np.empty((0, 7))