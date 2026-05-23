from screens import SETUP_BUTTON_RECT, point_in_rect


def test_point_in_rect_inside():
    x, y, w, h = SETUP_BUTTON_RECT
    assert point_in_rect(x + 5, y + 5, SETUP_BUTTON_RECT)
    assert point_in_rect(x + w // 2, y + h // 2, SETUP_BUTTON_RECT)


def test_point_in_rect_outside():
    x, y, w, h = SETUP_BUTTON_RECT
    assert not point_in_rect(x - 1, y - 1, SETUP_BUTTON_RECT)
    assert not point_in_rect(x + w + 1, y + h + 1, SETUP_BUTTON_RECT)
