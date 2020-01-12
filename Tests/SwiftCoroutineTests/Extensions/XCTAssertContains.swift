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

func XCTAssertEqual<T: Equatable>(_ expression1: Result<T, Error>?, _ expression2: T) {
    if let result = try? expression1?.get() {
        XCTAssertEqual(result, expression2)
    } else {
        XCTFail()
    }
}

func XCTAssertTrue(_ expression: Bool?) {
    if let expression = expression {
        XCTAssertTrue(expression)
    } else {
        XCTFail()
    }
}

func XCTAssertFalse(_ expression: Bool?) {
    if let expression = expression {
        XCTAssertFalse(expression)
    } else {
        XCTFail()
    }
}
