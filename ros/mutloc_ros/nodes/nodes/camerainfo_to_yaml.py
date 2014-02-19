
def camera_info_to_dict(camera_info):
    camdict = dict(image_width=int(camera_info.width),
                   image_height=int(camera_info.height),
                   camera_name="camera",
                   camera_matrix=dict(rows=3,
                                      cols=3,
                                      data=list(camera_info.K)),
                   distortion_model=camera_info.distortion_model,
                   distortion_coefficients=dict(rows=1,
                                                cols=5,
                                                data=list(camera_info.D)),
                   rectification_matrix=dict(rows=3,
                                             cols=3,
                                             data=list(camera_info.R)),
                   projection_matrix=dict(rows=3,
                                          cols=4,
                                          data=list(camera_info.P))
                  )
    return camdict
