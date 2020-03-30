//
//  CoFutureCancelTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureCancelTests: XCTestCase {
    
    func testCancel() {
        let exp = expectation(description: #function)
        exp.expectedFulfillmentCount = 4
        let future = CoPromise<Bool>()
        func test() {
            future.whenCanceled { exp.fulfill() }
            future.whenFailure { error in
                if let error = error as? CoFutureError {
                    XCTAssertEqual(error, .canceled)
                } else {
                    XCTFail()
                }
                exp.fulfill()
            }
        }
        test()
        XCTAssertFalse(future.isCanceled)
        future.cancel()
        test()
        XCTAssertTrue(future.isCanceled)
        wait(for: [exp], timeout: 1)
    }
    
    func testCancel2() {
        let future = CoPromise<Int>()
        let map = future.map { $0 + 1 }
        map.cancel()
        XCTAssertTrue(future.isCanceled)
        XCTAssertTrue(map.isCanceled)
    }
    
}
