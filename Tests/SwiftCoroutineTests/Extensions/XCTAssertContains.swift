//
//  XCTAssertContains.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import Foundation

func XCTAssertContains<T: Equatable>(_ value: T, in range: Range<T>) {
    XCTAssert(range.contains(value))
}

func XCTAssertDuration(from date: Date, in range: Range<Double>) {
    XCTAssertContains(Date().timeIntervalSince(date), in: range)
}
