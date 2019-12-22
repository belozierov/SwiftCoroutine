//
//  CoFutureTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine

class CoFutureTests: XCTestCase {
    
    func testTransform() {
        let expectation = XCTestExpectation(description: "Test Transform")
        expectation.expectedFulfillmentCount = 3
        let promise = async { () -> Int in
            sleep(1)
            return 1
        }
        var transformed: CoFuture<String>! = promise
            .map { $0 * 2 }
            .catch { _ in XCTFail() }
            .map { $0 * 3 }
            .then { XCTAssertEqual($0, 6) }
            .then { _ in expectation.fulfill() }
            .map { $0.description }
            .handler { expectation.fulfill() }
        weak var weakTransformed: CoFuture<String>? = transformed
        transformed.notify(queue: .global()) {
            transformed = nil
            XCTAssertNil(weakTransformed)
            XCTAssertEqual(try? $0.get(), "6")
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
    }
    
}
